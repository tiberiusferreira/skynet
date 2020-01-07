use crate::yolo_nn::network::original_yolo::config_file_parsing::{ConfigBlock, DarknetParsedFile};
use anyhow::{bail, ensure, Context};
use tch::nn::{ModuleT, VarStore};
use tch::{nn, Device, R4TensorGeneric, Tensor};

#[derive(Debug)]
pub struct DarknetConfig {
    pub width: u32,
    pub height: u32,
    pub layers: Vec<(i64, Bl)>,
    pub var_store: VarStore,
}

#[derive(Debug)]
pub enum Bl {
    Layer(Box<dyn ModuleT>),
    Route(Vec<usize>),
    Shortcut(usize),
    Yolo(i64, Vec<(i64, i64)>),
}

#[derive(Debug)]
pub struct YoloNetworkOutput {
    pub single_scale_output: Tensor,
    pub anchor_boxes: Vec<(i64, i64)>,
}

impl DarknetConfig {
    pub fn new(file_path: &str) -> Result<DarknetConfig, anyhow::Error> {
        let parsed_file = DarknetParsedFile::new(file_path)?;
        let width = parsed_file
            .get_global_params("width")
            .context("No width in net config")?
            .parse::<u32>()
            .context("Width in file was not a number")?;
        let height = parsed_file
            .get_global_params("height")
            .context("No height in net config")?
            .parse::<u32>()
            .context("Height in file was not a number")?;

        let var_store = VarStore::new(Device::Cpu);
        let root_path = var_store.root();

        let mut layers: Vec<(i64, Bl)> = vec![];
        // starts with RGB image, so 3 channels
        let mut prev_channels: i64 = 3;

        let mut iterator = parsed_file.config_blocks.iter().enumerate().peekable();
        while let Some((index, config_block)) = iterator.next() {
            let next_is_yolo;
            if let Some(true) = iterator.peek().map(|(_i, block)| block.block_type == "yolo"){
                next_is_yolo = true;
                println!("Next yolo");
            }else{
                next_is_yolo = false;
            }
            let (output_channels, bl) = match config_block.block_type.as_str() {
                "convolutional" => {
                    if next_is_yolo{
                        println!("Changed index {}", index);
                        conv(&(&root_path / index) / "custom", index, prev_channels, &config_block)?
//                        conv(&root_path / index, index, prev_channels, &config_block)?
                    }else{
                        conv(&root_path / index, index, prev_channels, &config_block)?
                    }
                },
                "upsample" => upsample(prev_channels)?,
                "shortcut" => shortcut(index, prev_channels, &config_block)?,
                "route" => route(index, &layers, &config_block)?,
                "yolo" => yolo(prev_channels, &config_block)?,
                otherwise => bail!("unsupported block type {}", otherwise),
            };
            prev_channels = output_channels;
            layers.push((output_channels, bl));
        }
//        for (index, config_block) in parsed_file.config_blocks.iter().enumerate() {
//            let (output_channels, bl) = match config_block.block_type.as_str() {
//                "convolutional" => conv(&root_path / index, index, prev_channels, &config_block)?,
//                "upsample" => upsample(prev_channels)?,
//                "shortcut" => shortcut(index, prev_channels, &config_block)?,
//                "route" => route(index, &layers, &config_block)?,
//                "yolo" => yolo(prev_channels, &config_block)?,
//                otherwise => bail!("unsupported block type {}", otherwise),
//            };
//            prev_channels = output_channels;
//            layers.push((output_channels, bl));
//        }

        Ok(DarknetConfig {
            width,
            height,
            layers,
            var_store,
        })
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn build_model(
        self,
    ) -> Result<
        (
            VarStore,
            Box<dyn Fn(&R4TensorGeneric, bool) -> Vec<YoloNetworkOutput>>,
        ),
        anyhow::Error,
    > {
        let image_height = self.height();
        let vs = self.var_store;
        let mut blocks: Vec<(i64, Bl)> = self.layers;
        let func = Box::new(
            move |n_3_416_416: &R4TensorGeneric, train: bool| -> Vec<YoloNetworkOutput> {
                let mut layer_outputs: Vec<Tensor> = vec![];
                let mut detections: Vec<YoloNetworkOutput> = vec![];
                let mut blocks_iter = blocks.iter().peekable();
                while let Some((_, b)) = blocks_iter.next() {
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
//                            println!("{:?}", anchors);
                            let last_output: R4TensorGeneric = layer_outputs
                                .last()
                                .unwrap_or(&n_3_416_416.tensor)
                                .shallow_clone()
                                .into();

                            //                        detections.push(detect(last_output, image_height, *classes, anchors));
                            /*
                            Originally indexed as (13x13 is an example) [1, 255, 13, 13]
                            which makes we index a given feature of the 255 possible features
                            3(number of bb)*85(features per bb (ex: objectness prob)) and then indexing the col and row for its value
                            We change it so we can index a row and col [13, 13] and get all the features
                            of the 3 bounding boxes for that location
                            */

                            //                        last_output.index(&[]);
                            let mut transposed: R4TensorGeneric = last_output
                                .tensor
                                .transpose(1, 2)
                                .transpose(2, 3) // (Batch_size, 13, 13, 255)
                                .contiguous()
                                .into();
                            let (n, w, h, feat_len) = transposed.tensor.size4().unwrap();
                            // From (Batch_size, 13, 13, 255) from the dimension 3, only select
                            // the 5th one onward, which are the objectness and classes probabilities
                            // elements (4, 5, ..)
                            let sigmoided = transposed.tensor.narrow(3, 4, feat_len - 4).sigmoid();
                            transposed.tensor.narrow(3, 4, feat_len - 4).copy_(&sigmoided);

                            // Also apply sigmoid to x and y of the BB elements (0 and 1)
                            let sigmoided = transposed.tensor.narrow(3, 0, 2).sigmoid();
                            transposed.tensor.narrow(3, 0, 2).copy_(&sigmoided);

//                            println!("last_output size = {:?}", transposed.tensor.size());
                            detections.push(YoloNetworkOutput {
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
        Ok((vs, func))
    }
}

fn conv(vs: nn::Path, index: usize, p: i64, b: &ConfigBlock) -> Result<(i64, Bl), anyhow::Error> {
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

fn upsample(prev_channels: i64) -> Result<(i64, Bl), anyhow::Error> {
    let layer = nn::func_t(|xs, _is_training| {
        let (_n, _c, h, w) = xs.size4().unwrap();
        xs.upsample_nearest2d(&[2 * h, 2 * w])
    });
    Ok((prev_channels, Bl::Layer(Box::new(layer))))
}

fn int_list_of_string(s: &str) -> Result<Vec<i64>, anyhow::Error> {
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

fn route(
    index: usize,
    p: &Vec<(i64, Bl)>,
    block: &ConfigBlock,
) -> Result<(i64, Bl), anyhow::Error> {
    let layers = int_list_of_string(block.get("layers")?)?;
    let layers: Vec<usize> = layers
        .into_iter()
        .map(|l| usize_of_index(index, l))
        .collect();
    let channels = layers.iter().map(|&l| p[l].0).sum();
    Ok((channels, Bl::Route(layers)))
}

fn shortcut(index: usize, p: i64, block: &ConfigBlock) -> Result<(i64, Bl), anyhow::Error> {
    let from = block.get("from")?.parse::<i64>()?;
    Ok((p, Bl::Shortcut(usize_of_index(index, from))))
}

fn yolo(p: i64, block: &ConfigBlock) -> Result<(i64, Bl), anyhow::Error> {
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
