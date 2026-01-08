#!/usr/bin/env python3
"""
Training script for the random feature diffusion model.

Implements gradient descent (SGD or Adam) dynamics on the learnable matrix A,
and tracks three types of losses:
    1. Train loss 
    2. Test loss 
    3. Error on the score 
"""

import os
import numpy as np
import torch
import torch.optim as optim
import argparse

main_dir = "results"
centroids_factor = 3.0
if centroids_factor == 1.0:
    end_dir = "partially_overlapped_centroids"
elif centroids_factor == 3.0:
    end_dir = "well_separated_centroids"


# ======================================================
#                   Argument Parser
# ======================================================
def get_args():
    parser = argparse.ArgumentParser(
        description="Training random feature diffusion model with class imbalance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ## parameters of the architecture and the dataset
    parser.add_argument("-psi_p", "--psi_p", type=float, default=2.0, help="ψ_p = p/d")
    parser.add_argument("-psi_n", "--psi_n", type=float, default=1.0, help="ψ_n = n/d")
    parser.add_argument("-d", "--d", type=int, default=int(2e3), help="data dimension")

    ## parameters of the diffusion
    parser.add_argument("-t", "--t", type=float, default=1e-2, help="Diffusion time")
    parser.add_argument("-diffusion_type", "--diffusion_type", choices=["FD", "OU"], default="OU",
                         help="Type of diffusion process: 'FD' (Free Diffusion) or 'OU' (Ornstein-Uhlenbeck)")

    ## parameters of the data distribution (gaussian mixture with isotropic clusters)
    parser.add_argument("-weight", "--weight", type=float, default=0.5, help="Class weight for the positive class")
    parser.add_argument("-var1", "--var1", type=float, default=0.5, help="Variance of the first cluster")
    parser.add_argument("-var2", "--var2", type=float, default=0.5, help="Variance of the second cluster")
    parser.add_argument("-r", "--r", type=float, default=0.0, help="Cosine similarity between the two cluster centers")

    ## training parameters
    parser.add_argument("-optimizer", "--optimizer", choices=["GD", "Adam"], default="GD", help="Optimizer type")
    parser.add_argument("-num_epochs", "--num_epochs", type=int, default=int(5e6), help="Number of training iterations")
    
    ## output
    parser.add_argument("-output_dir", "--output_dir", type=str, default=f"{main_dir}/results_training_RF/{end_dir}", help="Directory to save results")
    
    ## seed
    parser.add_argument("-seed", "--seed", type=int, default=42, help="Random seed for reproducibility")
    
    ## number of realizations
    parser.add_argument("-n_realizations", "--n_realizations", type=int, default=1, help="Number of independent training realizations")
    
    return parser.parse_args()


# ======================================================
#            Model Definition
# ======================================================
def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Derived parameters
    Psi_p, Psi_n, d, t = args.psi_p, args.psi_n, args.d, args.t
    diffusion_type = args.diffusion_type
    if diffusion_type == "FD":
        a_t = 1.0
        delta_t = t
    elif diffusion_type == "OU":
        a_t = np.exp(-t)
        delta_t = 1 - a_t**2
    p, n = int(Psi_p * d), int(Psi_n * d)
    weight, r = args.weight, args.r
    var_1 = args.var1
    var_2 = args.var2

    

    learning_rate = (2e-4) * d / delta_t # We use this scaling because the gradient is very small at initialization
    
    print(f"\n🚀 Training random feature diffusion model on imbalanced data")
    print(f"Device: {device}")
    print(f"Parameters: ψ_p={Psi_p}, ψ_n={Psi_n}, d={d}, t={t}")
    print(f"Optimizer: {args.optimizer},lr={learning_rate:.2e}")
    print(f"Number of epochs: {args.num_epochs:,}")
    print(f"Number of realizations: {args.n_realizations}\n")

    def generate_dataset(n, d, weight, vars, centroids_factor, r):
        
        #centroids_factor = 1.0

        n_1 = int(weight * n)
        n_2 = n - n_1

        var_1 = vars[0]
        var_2 = vars[1]

        ## spatially homogeneous centroids with given cosine similarity
        m_1 = centroids_factor * torch.ones(d, device=device)
        m_2 = -centroids_factor * torch.ones(d, device=device)
        if d % 2 == 0:
            ind = int(r * d // 2)
            m_2[d//2-ind:] *= -1.0
        else:
            ind = int(r * (d-1) // 2)
            m_2[d//2-ind:d//2+ind+1] *= -1.0


        ## check that the cosine similarity is correct
        #cos_sim = torch.dot(m_1, m_2) / (torch.norm(m_1) * torch.norm(m_2))
        #print(f"Cosine similarity between the two centroids: {cos_sim.item():.4f}")

        

        ## useful when we will generalized to structured clusters

        #cov_1 = var_1 * torch.eye(d, device=device)
        #cov_2 = var_2 * torch.eye(d, device=device)

        #cluster_1_samples = torch.randn(n_1, d, device=device) @ torch.linalg.cholesky(cov_1) + m_1
        #cluster_2_samples = torch.randn(n_2, d, device=device) @ torch.linalg.cholesky(cov_2) + m_2
        #dataset = torch.vstack([cluster_1_samples, cluster_2_samples])

        cluster_1_samples = torch.randn(n_1, d, device=device) * np.sqrt(var_1) + m_1
        cluster_2_samples = torch.randn(n_2, d, device=device) * np.sqrt(var_2) + m_2
        dataset = torch.vstack([cluster_1_samples, cluster_2_samples])

        return [dataset, m_1, m_2]

    # ------------------------------------------------------
    # Define S_A(x), here we use tanh as an activation function but it can be changed
    def model(x_input):
        return (A @ torch.tanh(W @ x_input.T / torch.sqrt(torch.tensor(d, device=device)))).T

    # ------------------------------------------------------
    # Loss function 
    def train_objective(A, x, eta):
        S_A_x = model(a_t * x + np.sqrt(delta_t) * eta)
        l2_term = torch.sum((np.sqrt(delta_t) * S_A_x + eta) ** 2) / (d * n)
        return l2_term 

    # ------------------------------------------------------
    # Train loss 
    def compute_train_loss(A, x, m=100):
        total_loss = 0
        with torch.no_grad():
            for _ in range(m):
                eta = torch.randn(n, d, device=device)
                S_A_x = model(a_t * x + np.sqrt(delta_t) * eta)
                loss = torch.sum((np.sqrt(delta_t) * S_A_x + eta) ** 2) / (d * n)
                total_loss += loss.item() / m
        return total_loss

    # ------------------------------------------------------
    # Test loss
    def compute_test_loss(A):
        batch_size = 100
        N = 100 * n
        test_dataset = generate_dataset(N, d, weight, [var_1, var_2], centroids_factor, r)[0]
        ## shuffle test_dataset to avoid any bias
        perm = torch.randperm(N)
        test_dataset = test_dataset[perm]
        eta = torch.randn(N, d, device=device)
        total_loss = 0.0
        num_batches = N // batch_size
        for i in range(num_batches):
            x_batch = test_dataset[i * batch_size : (i + 1) * batch_size]
            eta_batch = eta[i * batch_size : (i + 1) * batch_size]
            S_A_x = model(a_t * x_batch + np.sqrt(delta_t) * eta_batch)
            loss = torch.sum((np.sqrt(delta_t) * S_A_x + eta_batch) ** 2) / (d * N)
            total_loss += loss.item()
        return total_loss

    # ------------------------------------------------------
    # True score: functions (true score is -f({|x-m_c*a_t|**2}_c)*x+(weight*f_0({|x-m_c*a_t|**2}_c)*m_0+(1-weight)*f_1({|x-m_c*a_t|**2}_c)*m_1)*a_t)
    # Using logsumexp to avoid numerical overflow

    def f(ys, weight, vars):
        """Compute f = (w1*invG1*exp(E1) + w2*invG2*exp(E2)) / (w1*exp(E1) + w2*exp(E2))
        Using logsumexp for numerical stability."""
        y_1 = ys[0]
        y_2 = ys[1]
        
        y_1_sq_norm = torch.linalg.norm(y_1, axis=1, keepdims=True)**2
        y_2_sq_norm = torch.linalg.norm(y_2, axis=1, keepdims=True)**2

        var_1 = vars[0]
        var_2 = vars[1]

        inv_Gamma_1 = (var_1*a_t**2 + delta_t)**(-1)
        inv_Gamma_2 = (var_2*a_t**2 + delta_t)**(-1)

        # Log-space exponents
        log_ratio = d * torch.log(torch.tensor(inv_Gamma_2 / inv_Gamma_1, device=y_1.device))
        E_1 = -0.5 * y_1_sq_norm * inv_Gamma_1 + torch.log(torch.tensor(weight, device=y_1.device))
        E_2 = -0.5 * y_2_sq_norm * inv_Gamma_2 + log_ratio + torch.log(torch.tensor(1 - weight, device=y_1.device))

        # Stack for logsumexp: shape (batch, 2)
        log_weights_denom = torch.cat([E_1, E_2], dim=1)
        log_denom = torch.logsumexp(log_weights_denom, dim=1, keepdim=True)

        # Numerator: w1*invG1*exp(E1) + w2*invG2*exp(E2)
        log_num_1 = E_1 + torch.log(torch.tensor(inv_Gamma_1, device=y_1.device))
        log_num_2 = E_2 + torch.log(torch.tensor(inv_Gamma_2, device=y_1.device))
        log_weights_num = torch.cat([log_num_1, log_num_2], dim=1)
        log_num = torch.logsumexp(log_weights_num, dim=1, keepdim=True)

        return torch.exp(log_num - log_denom)

    def f_0(ys, weight, vars):
        """Compute f_0 = invG1*exp(E1) / (w1*exp(E1) + w2*exp(E2))
        Using logsumexp for numerical stability."""
        y_1 = ys[0]
        y_2 = ys[1]

        y_1_sq_norm = torch.linalg.norm(y_1, axis=1, keepdims=True)**2
        y_2_sq_norm = torch.linalg.norm(y_2, axis=1, keepdims=True)**2

        var_1 = vars[0]
        var_2 = vars[1]

        inv_Gamma_1 = (var_1*a_t**2 + delta_t)**(-1)
        inv_Gamma_2 = (var_2*a_t**2 + delta_t)**(-1)

        # Log-space exponents
        log_ratio = d * torch.log(torch.tensor(inv_Gamma_2 / inv_Gamma_1, device=y_1.device))
        E_1 = -0.5 * y_1_sq_norm * inv_Gamma_1 + torch.log(torch.tensor(weight, device=y_1.device))
        E_2 = -0.5 * y_2_sq_norm * inv_Gamma_2 + log_ratio + torch.log(torch.tensor(1 - weight, device=y_1.device))

        # Denominator via logsumexp
        log_weights_denom = torch.cat([E_1, E_2], dim=1)
        log_denom = torch.logsumexp(log_weights_denom, dim=1, keepdim=True)

        # Numerator: invG1*exp(-0.5*y1^2*invG1) (no weight factor)
        log_num = -0.5 * y_1_sq_norm * inv_Gamma_1 + torch.log(torch.tensor(inv_Gamma_1, device=y_1.device))

        return torch.exp(log_num - log_denom)
    
    def f_1(ys, weight, vars):
        """Compute f_1 = invG2*(invG2/invG1)^d*exp(E2_raw) / (w1*exp(E1) + w2*exp(E2))
        Using logsumexp for numerical stability."""
        y_1 = ys[0]
        y_2 = ys[1]

        y_1_sq_norm = torch.linalg.norm(y_1, axis=1, keepdims=True)**2
        y_2_sq_norm = torch.linalg.norm(y_2, axis=1, keepdims=True)**2

        var_1 = vars[0]
        var_2 = vars[1]

        inv_Gamma_1 = (var_1*a_t**2 + delta_t)**(-1)
        inv_Gamma_2 = (var_2*a_t**2 + delta_t)**(-1)

        # Log-space exponents
        log_ratio = d * torch.log(torch.tensor(inv_Gamma_2 / inv_Gamma_1, device=y_1.device))
        E_1 = -0.5 * y_1_sq_norm * inv_Gamma_1 + torch.log(torch.tensor(weight, device=y_1.device))
        E_2 = -0.5 * y_2_sq_norm * inv_Gamma_2 + log_ratio + torch.log(torch.tensor(1 - weight, device=y_1.device))

        # Denominator via logsumexp
        log_weights_denom = torch.cat([E_1, E_2], dim=1)
        log_denom = torch.logsumexp(log_weights_denom, dim=1, keepdim=True)

        # Numerator: invG2*(invG2/invG1)^d*exp(-0.5*y2^2*invG2) (no weight factor)
        log_num = -0.5 * y_2_sq_norm * inv_Gamma_2 + log_ratio + torch.log(torch.tensor(inv_Gamma_2, device=y_1.device))

        return torch.exp(log_num - log_denom)

    # ------------------------------------------------------
    # Error on the Score
    def compute_error_score(A):
        batch_size = 100
        N = 10_000
        inv_Gamma_sq_1 = (var_1*a_t**2+delta_t)**(-1)
        inv_Gamma_sq_2 = (var_2*a_t**2+delta_t)**(-1)
        #test_dataset = torch.randn(N, d, device=device)
        vars = [var_1*a_t**2+delta_t, var_2*a_t**2+delta_t]
        test_dataset, m_1, m_2 = generate_dataset(N, d, weight, vars, centroids_factor*a_t, r)
        ## shuffle test_dataset to avoid any bias
        perm = torch.randperm(N)
        test_dataset = test_dataset[perm]
        total_loss = 0.0
        total_loss_cluster_1 = 0.0
        total_loss_cluster_2 = 0.0
        num_batches = N // batch_size
        for i in range(num_batches):
            x_batch = test_dataset[i * batch_size : (i + 1) * batch_size]
            S_A_x = model(x_batch)
            y1 = x_batch - torch.outer(torch.ones(batch_size, device=device), m_1)
            y2 = x_batch - torch.outer(torch.ones(batch_size, device=device), m_2)
            ys = [y1, y2]
            f_vals = f(ys, weight, vars)
            f0_vals = f_0(ys, weight, vars)
            f1_vals = f_1(ys, weight, vars)
            true_score = -f_vals * x_batch + (weight * f0_vals * torch.outer(torch.ones(batch_size, device=device), m_1) + (1 - weight) * f1_vals * torch.outer(torch.ones(batch_size, device=device), m_2))
            loss = torch.sum((S_A_x - true_score) ** 2) / (d * N)
            loss_cluster_1 = torch.sum((S_A_x+y1*inv_Gamma_sq_1)**2)/(d * N)
            loss_cluster_2 = torch.sum((S_A_x+y2*inv_Gamma_sq_2)**2)/(d * N)
            total_loss += loss.item()
            total_loss_cluster_1 += loss_cluster_1.item()
            total_loss_cluster_2 += loss_cluster_2.item()
        return total_loss, total_loss_cluster_1, total_loss_cluster_2

    # ------------------------------------------------------
    # Logarithmic sampling of epochs
    def sample_logarithmically(N, T):
        log_samples = np.linspace(np.log(1), np.log(T), N)
        samples = np.exp(log_samples)
        return sorted({int(v) for v in samples})

    sample_times = sample_logarithmically(200, args.num_epochs)

    # ======================================================
    #                    Output Setup
    # ======================================================
    run_name = f"Psi_p={Psi_p:.1f}_Psi_n={Psi_n:.1f}_d={d:.0e}_t={t:.0e}_{args.optimizer}"
    save_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving results to: {save_dir}\n")

    # ======================================================
    #                    Realization Loop
    # ======================================================
    for realization_idx in range(args.n_realizations):
        print(f"\n{'='*80}")
        print(f"[Realization {realization_idx + 1}/{args.n_realizations}]")
        print(f"{'='*80}")
        
        # Set seed for this realization (different for each)
        realization_seed = args.seed + realization_idx
        torch.manual_seed(realization_seed)
        np.random.seed(realization_seed)
        
        # Re-initialize model components for each realization
        W = torch.randn(p, d, device=device)
        A = torch.zeros(d, p, requires_grad=True, device=device)
        
        # Re-generate dataset for each realization
        x, m_1, m_2 = generate_dataset(n, d, weight, [var_1, var_2], centroids_factor, r)
        
        # Re-initialize optimizer
        optimizer = optim.SGD([A], lr=learning_rate) if args.optimizer == "GD" else optim.Adam([A], lr=learning_rate)
        
        # Arrays for this realization
        Train_loss, Test_loss, Error_score, Error_score_1, Error_score_2, Training_time = [], [], [], [], [], []

        for epoch in range(args.num_epochs):
            eta = torch.randn(n, d, device=device)
            loss = train_objective(A, x, eta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging at sampled times
            if epoch in sample_times:
                with torch.no_grad():
                    train = compute_train_loss(A, x)
                    test = compute_test_loss(A)
                    error, error_1, error_2 = compute_error_score(A)

                    Train_loss.append(train)
                    Test_loss.append(test)
                    Error_score.append(error)
                    Error_score_1.append(error_1)
                    Error_score_2.append(error_2)
                    Training_time.append(epoch * learning_rate)

                print(f"[Epoch {epoch:,}] Train={train:.4e} | Test={test:.4e} | Error on the score={error:.4e} | Error on cluster 1={error_1:.4e} | Error on cluster 2={error_2:.4e}")

        # After training, append results to files (load existing, append, save)
        def append_to_npy(filepath, new_data):
            """Load existing .npy file if exists, append new data as new row, save back."""
            new_data = np.array(new_data).reshape(1, -1)  # Shape: (1, n_samples)
            if os.path.exists(filepath):
                existing = np.load(filepath)
                combined = np.vstack([existing, new_data])
            else:
                combined = new_data
            np.save(filepath, combined)
        
        append_to_npy(os.path.join(save_dir, f"Train_loss_weight={weight}_var1={var_1}_var2={var_2}.npy"), Train_loss)
        append_to_npy(os.path.join(save_dir, f"Test_loss_weight={weight}_var1={var_1}_var2={var_2}.npy"), Test_loss)
        append_to_npy(os.path.join(save_dir, f"Error_score_weight={weight}_var1={var_1}_var2={var_2}.npy"), Error_score)
        append_to_npy(os.path.join(save_dir, f"Training_time_weight={weight}_var1={var_1}_var2={var_2}.npy"), Training_time)
        append_to_npy(os.path.join(save_dir, f"Error_score_1_weight={weight}_var1={var_1}_var2={var_2}.npy"), Error_score_1)
        append_to_npy(os.path.join(save_dir, f"Error_score_2_weight={weight}_var1={var_1}_var2={var_2}.npy"), Error_score_2)
        
        print(f"\n✅ Realization {realization_idx + 1} complete. Results appended.")

    print(f"\n{'='*80}")
    print(f"✅ All {args.n_realizations} realizations complete. Results saved to: {save_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()




