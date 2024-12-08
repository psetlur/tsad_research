from .encoder import Encoder


def load_model(config, a_config):
    return Encoder(
            ts_input_size=config.get("ts_input_size"),
            lr=config.get("lr"),
            a_config=a_config
        )