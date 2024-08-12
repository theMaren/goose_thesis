import argparse
import os
import time
import torch
from datetime import datetime
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from models.gnn.loss import MSELoss
from models.gnn.train_eval import train, evaluate
from models.save_load import load_gnn_model,save_gnn_model_from_dict
from dataset.dataset_gnn import get_tensor_graphs_from_plans

def setup_arg_parser():
    """Setup argument parsing."""
    parser = argparse.ArgumentParser(description='Train GNN models on planning problems.')
    parser.add_argument('--model', required=True, help='Path to the model file.')
    parser.add_argument('--domain', required=True, help='Path to the domain PDDL file.')
    parser.add_argument('--difficulty', required=True, help='The graph representation used.')
    return parser.parse_args()

def prepare_data_loaders(dataset, batch_size):
    """Prepare training and validation data loaders."""
    train_set, val_set = train_test_split(dataset, test_size=0.15, random_state=4550)
    print("train size:", len(train_set))
    print("validation size:", len(val_set))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def write_retrain_log(model_name, epochs, num_plans, time_total, time_retrain, time_graph, train_loss ,val_loss, combined_loss, train_loss_0, val_loss_0,combined_loss_0):
    # Get the current timestamp
    model_name_without_extension, _ = os.path.splitext(model_name)
    current_time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Create the filename
    filename = f'retraining/retrain_logs/retrain_{model_name_without_extension}.txt'

    # Create the content to be written to the file
    content = (
        f'Retrain {model_name} finished at {current_time_stamp}\n'
        f'epochs: {epochs}\n'
        f'number of problems retrained: {num_plans}\n'
        f'total time: {time_total}\n'
        f'retrain time: {time_retrain}\n'
        f'graph generation time: {time_graph}\n'
        f'train_loss_before: {train_loss_0}\n' #loss of epoch 0 (so before any retraining is done) 
        f'val_loss_before: {val_loss_0}\n' #loss of epoch 0 (so before any retraining is done) 
        f'combined_loss_before: {combined_loss_0}\n' #loss of epoch 0 
        f'train_loss_after: {train_loss}\n'
        f'val_loss_after: {val_loss}\n'
        f'combined_loss_after: {combined_loss}\n'
        '\n'
    )

    # Write or append the content to the file
    with open(filename, 'a') as file:
        file.write(content)

def count_files_in_directory(directory_path):
    # List all entries in the directory
    entries = os.listdir(directory_path)
    # Filter out directories, only keep files
    files = [entry for entry in entries if os.path.isfile(os.path.join(directory_path, entry))]
    # Count the number of files
    return len(files)


def main():
    args = setup_arg_parser()
    args.rep = "ilg"
    args.planner = "fd"
    #args.device = 0

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    args.device = device

    current_directory = os.getcwd()

    args.plans_dir = os.path.join(current_directory, f"retraining/solutions/{args.difficulty}/{args.domain}")
    args.domain_pddl = os.path.join(current_directory, f"benchmarks/ipc23-learning/{args.domain}/domain.pddl")
    args.tasks_dir = os.path.join(current_directory, f"retraining/dataset_new/{args.domain}/testing_new/{args.difficulty}")

    print(f"Device: {device}")
    start_time_total = time.time()
    start_time_graphs = time.time()
    
    num_plans = count_files_in_directory(args.plans_dir)
    dataset_retrain = get_tensor_graphs_from_plans(args)

    #if retrain on hard include the medium problems
    if args.difficulty == "hard":
        args.plans_dir = os.path.join(current_directory, f"retraining/solutions/medium/{args.domain}")
        args.tasks_dir = os.path.join(current_directory, f"retraining/dataset_new/{args.domain}/testing_new/medium")
        num_plans = num_plans + count_files_in_directory(args.plans_dir)
        dataset_medium = get_tensor_graphs_from_plans(args)

    #include the easy problems
    args.plans_dir = os.path.join(current_directory, f"benchmarks/ipc23-learning/{args.domain}/training_plans")
    args.tasks_dir = os.path.join(current_directory, f"benchmarks/ipc23-learning/{args.domain}/training/easy")

    num_plans = num_plans + count_files_in_directory(args.plans_dir)
    dataset_easy = get_tensor_graphs_from_plans(args)

    if args.difficulty == "hard":
        dataset = dataset_retrain + dataset_medium + dataset_easy
    else:
        dataset = dataset_retrain + dataset_easy


    train_loader, val_loader = prepare_data_loaders(dataset, batch_size=16)
    graph_time = time.time() - start_time_graphs #take time for graph generation
    
    # Load model
    os.chdir(os.path.join(os.getcwd(), os.pardir))  # Move up one directory
    print("Current working directory:", os.getcwd())
    model, model_args = load_gnn_model(args.model)
    model.device = device
    model.model.device = device
    model_args.device = device
    print("Model loaded")

    # Initialize optimizer and loss criterion
    criterion = MSELoss()
    optimiser = Adam(model.parameters(), lr=model_args.lr)
    scheduler = ReduceLROnPlateau(optimiser, mode="min", verbose=True, factor=model_args.reduction, patience=model_args.patience)

    # Training loop
    best_dict, best_metric, best_epoch = None, float("inf"), 0
    print("Retraining...")
    try:
        start_time_retrain = time.time()
        for e in range(model_args.epochs):
            t = time.time()

            train_stats = train(model, device, train_loader, criterion, optimiser)
            val_stats = evaluate(model, device, val_loader, criterion)
            scheduler.step(val_stats["loss"])
            
            #save metrics before retraining
            if e == 0:
                train_loss_0 = train_stats["loss"]
                val_loss_0 = val_stats["loss"]
                combined_loss_0 = (train_stats["loss"] + 2 * val_stats["loss"]) / 3

            # Update best model if improved
            combined_metric = (train_stats["loss"] + 2 * val_stats["loss"]) / 3
            if combined_metric < best_metric:
                best_train_loss = train_stats["loss"]
                best_val_loss = val_stats["loss"]
                best_metric = combined_metric
                best_dict = model.model.state_dict()
                best_epoch = e
                
            print(f"Epoch {e}, Time {time.time() - t:.1f}s, Train Loss {train_stats['loss']:.2f}, Val Loss {val_stats['loss']:.2f}")

            # Check for early stopping condition
            lr = optimiser.param_groups[0]["lr"]
            if lr < 1e-5:
                print(f"Early stopping due to small learning rate: {lr}")
                break

    except KeyboardInterrupt:
        print("Training interrupted by user.")

    time_retrain = time.time() - start_time_retrain
    print("Training completed. Best epoch:", best_epoch)

    model_args.save_file = args.model
    #save teh retrained model
    model_args.best_metric = best_metric
    save_gnn_model_from_dict(best_dict, model_args)
    time_total = time.time() - start_time_total
    print("Model_saved")

    #num_plans = count_files_in_directory(args.plans_dir)
    write_retrain_log(os.path.basename(args.model), e, num_plans ,time_total, time_retrain, graph_time, best_train_loss, best_val_loss, best_metric, train_loss_0, val_loss_0, combined_loss_0)


if __name__ == "__main__":
    main()
