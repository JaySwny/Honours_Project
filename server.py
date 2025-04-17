import flwr as fl
from config import NUM_ROUNDS
from config import NUM_CLIENTS

def fit_config(rnd: int):
    return {"round": rnd}

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    min_available_clients=NUM_CLIENTS,
    fraction_evaluate=1.0,
    min_fit_clients=NUM_CLIENTS,
    min_evaluate_clients=NUM_CLIENTS,
    on_fit_config_fn=fit_config

)

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy
)