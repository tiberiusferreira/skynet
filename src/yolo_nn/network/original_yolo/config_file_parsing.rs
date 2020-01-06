use anyhow::{bail, ensure, Context};
use futures::io::{Error, ErrorKind};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug)]
pub struct DarknetParsedFile {
    pub config_blocks: Vec<ConfigBlock>,
    pub global_net_params: HashMap<String, String>,
}

impl DarknetParsedFile {
    pub fn new<T: AsRef<Path>>(path: T) -> Result<DarknetParsedFile, anyhow::Error> {
        let file = File::open(path.as_ref())?;
        let mut builder = DarknetConfigBuilder::new();
        for line in BufReader::new(file).lines() {
            let line = line?;
            // # Means comment
            if line.is_empty() || line.starts_with("#") {
                continue;
            }
            let line = line.trim();
            // start of a block
            if line.starts_with("[") {
                ensure!(
                    line.ends_with("]"),
                    "block declaration line does not end with ']' {}",
                    line
                );
                // Block name
                let line_contents = &line[1..line.len() - 1];
                // finish the previous block since we are starting a new one
                builder.finish_current_block();
                builder.block_type_begin_parsed = Some(line_contents.to_string());
            } else {
                // parameters of the current block
                let key_value: Vec<&str> = line.splitn(2, "=").collect();
                ensure!(key_value.len() == 2, "missing equal {}", line);
                let prev = builder.parameters_of_block_being_parsed.insert(
                    key_value[0].trim().to_owned(),
                    key_value[1].trim().to_owned(),
                );
                ensure!(prev == None, "multiple value for key {}", line);
            }
        }
        builder.finish_current_block();
        Ok(builder.darknet_config)
    }

    pub fn get_global_params(&self, key: &str) -> Result<&str, anyhow::Error> {
        match self.global_net_params.get(&key.to_string()) {
            None => bail!("cannot find {} in net parameters", key),
            Some(value) => Ok(value),
        }
    }
}

struct DarknetConfigBuilder {
    block_type_begin_parsed: Option<String>,
    parameters_of_block_being_parsed: HashMap<String, String>,
    darknet_config: DarknetParsedFile,
}

impl DarknetConfigBuilder {
    fn new() -> DarknetConfigBuilder {
        DarknetConfigBuilder {
            block_type_begin_parsed: None,
            parameters_of_block_being_parsed: HashMap::new(),
            darknet_config: DarknetParsedFile {
                config_blocks: vec![],
                global_net_params: HashMap::new(),
            },
        }
    }

    fn finish_current_block(&mut self) {
        match &self.block_type_begin_parsed {
            // None if we just started parsing the file
            None => (),
            Some(block_type) => {
                if block_type == "net" {
                    self.darknet_config.global_net_params =
                        self.parameters_of_block_being_parsed.clone();
                } else {
                    let new_block_conf = ConfigBlock {
                        block_type: block_type.to_string(),
                        parameters: self.parameters_of_block_being_parsed.clone(),
                    };
                    self.darknet_config.config_blocks.push(new_block_conf);
                }
            }
        }
        // Clear current block data
        self.block_type_begin_parsed = None;
        self.parameters_of_block_being_parsed.clear();
    }
}

#[derive(Debug)]
pub struct ConfigBlock {
    pub block_type: String,
    pub parameters: HashMap<String, String>,
}

impl ConfigBlock {
    pub fn get(&self, key: &str) -> Result<&str, anyhow::Error> {
        match self.parameters.get(&key.to_string()) {
            None => bail!("cannot find {} in {}", key, self.block_type),
            Some(value) => Ok(value),
        }
    }
}
