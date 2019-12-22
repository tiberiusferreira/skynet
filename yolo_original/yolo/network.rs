use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::exit;
use tch::{nn, nn::{FuncT, ModuleT}, Device, Kind, R1Tensor, R2Tensor, R4Tensor, Tensor, R4TensorGeneric};

#[derive(Debug)]
struct Block {
    block_type: String,
    parameters: BTreeMap<String, String>,
}

impl Block {
    fn get(&self, key: &str) -> failure::Fallible<&str> {
        match self.parameters.get(&key.to_string()) {
            None => bail!("cannot find {} in {}", key, self.block_type),
            Some(value) => Ok(value),
        }
    }
}

#[derive(Debug)]
pub struct Darknet {
    blocks: Vec<Block>,
    parameters: BTreeMap<String, String>,
}

impl Darknet {
    fn get(&self, key: &str) -> failure::Fallible<&str> {
        match self.parameters.get(&key.to_string()) {
            None => bail!("cannot find {} in net parameters", key),
            Some(value) => Ok(value),
        }
    }
}

struct Accumulator {
    block_type: Option<String>,
    parameters: BTreeMap<String, String>,
    net: Darknet,
}

impl Accumulator {
    fn new() -> Accumulator {
        Accumulator {
            block_type: None,
            parameters: BTreeMap::new(),
            net: Darknet {
                blocks: vec![],
                parameters: BTreeMap::new(),
            },
        }
    }

    fn finish_block(&mut self) {
        match &self.block_type {
            None => (),
            Some(block_type) => {
                if block_type == "net" {
                    self.net.parameters = self.parameters.clone();
                } else {
                    let block = Block {
                        block_type: block_type.to_string(),
                        parameters: self.parameters.clone(),
                    };
                    self.net.blocks.push(block);
                }
                self.parameters.clear();
            }
        }
        self.block_type = None;
    }
}

pub fn parse_config<T: AsRef<Path>>(path: T) -> failure::Fallible<Darknet> {
    let file = File::open(path.as_ref())?;
    let mut acc = Accumulator::new();
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.is_empty() || line.starts_with("#") {
            continue;
        }
        let line = line.trim();
        if line.starts_with("[") {
            ensure!(line.ends_with("]"), "line does not end with ']' {}", line);
            let line = &line[1..line.len() - 1];
            acc.finish_block();
            acc.block_type = Some(line.to_string());
        } else {
            let key_value: Vec<&str> = line.splitn(2, "=").collect();
            ensure!(key_value.len() == 2, "missing equal {}", line);
            let prev = acc.parameters.insert(
                key_value[0].trim().to_owned(),
                key_value[1].trim().to_owned(),
            );
            ensure!(prev == None, "multiple value for key {}", line);
        }
    }
    acc.finish_block();
    Ok(acc.net)
}

enum Bl {
    Layer(Box<dyn ModuleT>),
    Route(Vec<usize>),
    Shortcut(usize),
    Yolo(i64, Vec<(i64, i64)>),
}

fn conv(vs: nn::Path, index: usize, p: i64, b: &Block) -> failure::Fallible<(i64, Bl)> {
    let activation = b.get("activation")?;
    let filters = b.get("filters")?.parse::<i64>()?;
    let pad = b.get("pad")?.parse::<i64>()?;
    let size = b.get("size")?.parse::<i64>()?;
    let stride = b.get("stride")?.parse::<i64>()?;
    let pad = if pad != 0 { (size - 1) / 2 } else { 0 };
    let (bn, bias) = match b.parameters.get("batch_normalize") {
        Some(p) if p.parse::<i64>()? != 0 => {
            let vs = &vs / format!("batch_norm_{}", index);
            let bn = nn::batch_norm2d(&vs, filters, Default::default());
            (Some(bn), false)
        }
        Some(_) | None => (None, true),
    };
    let conv_cfg = nn::ConvConfig {
        stride,
        padding: pad,
        bias,
        ..Default::default()
    };
    let vs = &vs / format!("conv_{}", index);
    let conv = nn::conv2d(vs, p, filters, size, conv_cfg);
    let leaky = match activation {
        "leaky" => true,
        "linear" => false,
        otherwise => bail!("unsupported activation {}", otherwise),
    };
    let func = nn::func_t(move |xs, train| {
        let xs = xs.apply(&conv);
        let xs = match &bn {
            Some(bn) => xs.apply_t(bn, train),
            None => xs,
        };
        if leaky {
            xs.max1(&(&xs * 0.1))
        } else {
            xs
        }
    });
    Ok((filters, Bl::Layer(Box::new(func))))
}

fn upsample(prev_channels: i64) -> failure::Fallible<(i64, Bl)> {
    let layer = nn::func_t(|xs, _is_training| {
        let (_n, _c, h, w) = xs.size4().unwrap();
        xs.upsample_nearest2d(&[2 * h, 2 * w])
    });
    Ok((prev_channels, Bl::Layer(Box::new(layer))))
}

fn int_list_of_string(s: &str) -> failure::Fallible<Vec<i64>> {
    let res: Result<Vec<_>, _> = s.split(",").map(|xs| xs.trim().parse::<i64>()).collect();
    Ok(res?)
}

fn usize_of_index(index: usize, i: i64) -> usize {
    if i >= 0 {
        i as usize
    } else {
        (index as i64 + i) as usize
    }
}

fn route(index: usize, p: &Vec<(i64, Bl)>, block: &Block) -> failure::Fallible<(i64, Bl)> {
    let layers = int_list_of_string(block.get("layers")?)?;
    let layers: Vec<usize> = layers
        .into_iter()
        .map(|l| usize_of_index(index, l))
        .collect();
    let channels = layers.iter().map(|&l| p[l].0).sum();
    Ok((channels, Bl::Route(layers)))
}

fn shortcut(index: usize, p: i64, block: &Block) -> failure::Fallible<(i64, Bl)> {
    let from = block.get("from")?.parse::<i64>()?;
    Ok((p, Bl::Shortcut(usize_of_index(index, from))))
}

fn yolo(p: i64, block: &Block) -> failure::Fallible<(i64, Bl)> {
    let classes = block.get("classes")?.parse::<i64>()?;
    let flat = int_list_of_string(block.get("anchors")?)?;
    ensure!(flat.len() % 2 == 0, "even number of anchors");
    let anchors: Vec<_> = (0..(flat.len() / 2))
        .map(|i| (flat[2 * i], flat[2 * i + 1]))
        .collect();
    let mask = int_list_of_string(block.get("mask")?)?;
    let anchors = mask.into_iter().map(|i| anchors[i as usize]).collect();
    Ok((p, Bl::Yolo(classes, anchors)))
}

// Apply f to a slice of tensor xs and replace xs values with f output.
fn slice_apply_and_set<F>(xs: &mut Tensor, start: i64, len: i64, f: F)
where
    F: FnOnce(&Tensor) -> Tensor,
{
    let mut slice = xs.narrow(2, start, len);
    let src = f(&slice);
    slice.copy_(&src)
}

fn detect(xs: &Tensor, image_height: i64, classes: i64, anchors: &Vec<(i64, i64)>) -> Tensor {
    let (bsize, _channels, conv_output_height, _width) = xs.size4().unwrap();
    let img_height_per_conv_output_height = image_height / conv_output_height;
    let grid_size = conv_output_height;
    // println!("{:?}  {:?}  {:?}  {:?}", image_height, conv_output_height, img_height_per_conv_output_height, grid_size);
    let nb_attrs_per_bb = 5 + classes;
    let nb_anchors_per_bb = anchors.len() as i64;
    let mut xs = xs //  (1, 255, 13, 13)
        .view((
            bsize,                               // 1
            nb_attrs_per_bb * nb_anchors_per_bb, // 85 * 3 = 255
            grid_size * grid_size,               // 13 * 13
        ))
        .transpose(1, 2) // (1, 13*13, 255)
        .contiguous()
        .view((
            bsize,
            grid_size * grid_size * nb_anchors_per_bb,
            nb_attrs_per_bb,
        )); // (1, 13*13*3, 85)
    let grid = Tensor::arange(grid_size, tch::kind::FLOAT_CPU);
    let a = grid.repeat(&[grid_size, 1]);
    // a = 0 1 2 ..
    //     0 1 2 ..
    //     0 1 2 ..
    let b = a.tr().contiguous();
    // b = 0 0 0 ..
    //     1 1 1 ..
    //     2 2 2 ..
    let x_offset = a.view((-1, 1));
    let y_offset = b.view((-1, 1));
    let xy_offset = Tensor::cat(&[x_offset, y_offset], 1)
        .repeat(&[1, nb_anchors_per_bb])
        .view((-1, 2))
        .unsqueeze(0);
    let anchors: Vec<f32> = anchors
        .iter()
        .flat_map(|&(x, y)| {
            vec![
                x as f32 / img_height_per_conv_output_height as f32,
                y as f32 / img_height_per_conv_output_height as f32,
            ]
            .into_iter()
        })
        .collect();
    let anchors = Tensor::of_slice(&anchors)
        .view((-1, 2))
        .repeat(&[grid_size * grid_size, 1])
        .unsqueeze(0);
    // xs here is: [box_coordinates objectness_score class_scores]
    // [tx, ty, tw, th,         po,         p1, p2, ...]
    // box_x, box_y, box_w, box_h = box coordinates in grid coordinates
    // box_x = sigmoid(tx) + c_x (x of top left of the grid cell)
    // box_y = sigmoid(ty) + c_y (y of top left of the grid cell)
    slice_apply_and_set(&mut xs, 0, 2, |xs| xs.sigmoid() + xy_offset);
    // first element is objectness probability, others are classes probabilities
    slice_apply_and_set(&mut xs, 4, 1 + classes, Tensor::sigmoid);
    // box_w = p_w*e^(t_w)
    // box_h = p_h*e^(t_h)
    slice_apply_and_set(&mut xs, 2, 2, |xs| xs.exp() * anchors);
    // Converting coordinates from grid (13x13 for example) to actual image size
    slice_apply_and_set(&mut xs, 0, 4, |xs| xs * img_height_per_conv_output_height);
    xs
}

pub struct NetworkOutput {
    pub single_scale_output: Tensor,
    pub anchor_boxes: Vec<(i64, i64)>,
}

impl Darknet {
    pub fn height(&self) -> failure::Fallible<i64> {
        let image_height = self.get("height")?.parse::<i64>()?;
        Ok(image_height)
    }

    pub fn width(&self) -> failure::Fallible<i64> {
        let image_width = self.get("width")?.parse::<i64>()?;
        Ok(image_width)
    }

    pub fn build_model(
        &self,
        vs: &nn::Path,
    ) -> failure::Fallible<Box<dyn Fn(&R4TensorGeneric, bool) -> Vec<NetworkOutput>>> {
        let mut blocks: Vec<(i64, Bl)> = vec![];
        let mut prev_channels: i64 = 3;
        for (index, block) in self.blocks.iter().enumerate() {
            let channels_and_bl = match block.block_type.as_str() {
                "convolutional" => conv(vs / index, index, prev_channels, &block)?,
                "upsample" => upsample(prev_channels)?,
                "shortcut" => shortcut(index, prev_channels, &block)?,
                "route" => route(index, &blocks, &block)?,
                "yolo" => yolo(prev_channels, &block)?,
                otherwise => bail!("unsupported block type {}", otherwise),
            };
            prev_channels = channels_and_bl.0;
            blocks.push(channels_and_bl);
        }
        let image_height = self.height()?;
        let func = Box::new(
            move |n_3_416_416: &R4TensorGeneric, train: bool| -> Vec<NetworkOutput> {
                let mut layer_outputs: Vec<Tensor> = vec![];
                let mut detections: Vec<NetworkOutput> = vec![];
                for (_, b) in blocks.iter() {
                    let ys = match b {
                        Bl::Layer(l) => {
                            let xs = layer_outputs.last().unwrap_or(&n_3_416_416.tensor);
                            l.forward_t(&xs, train)
                        }
                        Bl::Route(layers) => {
                            let layers: Vec<_> =
                                layers.iter().map(|&i| &layer_outputs[i]).collect();
                            Tensor::cat(&layers, 1)
                        }
                        Bl::Shortcut(from) => {
                            layer_outputs.last().unwrap() + layer_outputs.get(*from).unwrap()
                        }
                        Bl::Yolo(classes, anchors) => {
                            println!("{:?}", anchors);
                            let last_output: R4TensorGeneric =
                                layer_outputs.last().unwrap_or(&n_3_416_416.tensor).shallow_clone().into();

                            println!("last_output size = {:?}", last_output.tensor.size());
                            //                        detections.push(detect(last_output, image_height, *classes, anchors));
                            /*
                            Originally indexed as (13x13 is an example) [1, 255, 13, 13]
                            which makes we index a given feature of the 255 possible features
                            3(number of bb)*85(features per bb (ex: objectness prob)) and then indexing the col and row for its value
                            We change it so we can index a row and col [13, 13] and get all the features
                            of the 3 bounding boxes for that location
                            */

                            //                        last_output.index(&[]);
                            let transposed: R4TensorGeneric = last_output.tensor
                                .transpose(1, 2)
                                .transpose(2, 3) // (Batch_size, 13, 13, 255)
                                .contiguous().into();
                            let (n, w, h, feat_len) = transposed.tensor.size4().unwrap();
                            // From (Batch_size, 13, 13, 255) from the dimension 3, only select
                            // the 5th one onward, which are the objectness and classes probabilities
                            // elements (4, 5, ..)
                            transposed.tensor.narrow(3, 4, feat_len - 4).sigmoid_();
                            // Also apply sigmoid to x and y of the BB elements (0 and 1)
                            transposed.tensor.narrow(3, 0, 2).sigmoid_();
                            detections.push(NetworkOutput {
                                single_scale_output: transposed.tensor,
                                anchor_boxes: anchors.to_vec(),
                            });
                            Tensor::default()
                        }
                    };
                    layer_outputs.push(ys);
                }
                detections
            },
        );
        Ok(func)
    }
}
