import nemo
import pytorch_lightning as pl
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf

try:
    from ruamel.yaml import YAML
except ModuleNotFoundError:
    from ruamel_yaml import YAML



train_manifest = "./dataset/train.json"
test_manifest = "./dataset/val.json"

vocab_file = "./nemo_experiment/tokenizer_spe_bpe_v1024"
config_file = "./configs/fast-conformer_ctc_bpe_bangla.yaml"


yaml = YAML(typ='safe')
with open(config_file) as f:
    params = yaml.load(f)
print(params)


trainer = pl.Trainer(devices=1, accelerator='gpu', max_epochs=1000)

params['model']['train_ds']['manifest_filepath'] = train_manifest
params['model']['validation_ds']['manifest_filepath'] = test_manifest
params['model']['tokenizer']['dir'] = vocab_file
#EncDecCTCModelBPE

type(params)

# Load the configuration file using OmegaConf
# config_path = './configs/config_conformer.yaml'
# config = OmegaConf.load(config_path)
config = OmegaConf.create(params)

# Create the Conformer CTC model
model = nemo_asr.models.EncDecCTCModelBPE(cfg=config.model, trainer=trainer)
trainer.fit(model)