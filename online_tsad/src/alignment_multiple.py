def black_box_function(args, model, train_dataloader, val_dataloader, test_dataloader, a_config):
    from alignment import EmbNormalizer, inject
    import numpy as np
    import torch
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import random
    import geomloss
    from matplotlib.lines import Line2D
    import matplotlib.cm as cm
    import os
    from collections import defaultdict

    ratio_anomaly = 0.1
    train_levels = np.round(np.arange(-1.0, 1.1, 0.1), 1)
    train_lengths = np.round(np.arange(0.2, 0.52, 0.02), 2)
    valid_levels = np.round(np.arange(-1.0, 1.1, 0.1), 1)
    valid_lengths = np.round(np.arange(0.2, 0.52, 0.02), 2)
    
    # using a subset for testing
    mixed_levels = np.round(np.arange(-0.5, 0.6, 0.5), 1)  # [-0.5, 0.0, 0.5]
    mixed_lengths = np.round(np.arange(0.2, 0.5, 0.1), 2)   # [0.2, 0.3, 0.4]

    anomaly_types = ['platform', 'mean']
    
    with torch.no_grad():
        # Get embeddings for all data
        z_train, x_train_np = [], []
        for x_batch in train_dataloader:
            c_x = model(x_batch.to(args.device)).detach()
            z_train.append(c_x)
            x_train_np.append(x_batch.numpy())
        z_train = torch.cat(z_train, dim=0)
        x_train_np = np.concatenate(x_train_np, axis=0).reshape(len(z_train), -1)

        z_valid, x_valid_np = [], []
        for x_batch in val_dataloader:
            c_x = model(x_batch.to(args.device)).detach()
            z_valid.append(c_x)
            x_valid_np.append(x_batch.numpy())
        z_valid = torch.cat(z_valid, dim=0)
        x_valid_np = np.concatenate(x_valid_np, axis=0).reshape(len(z_valid), -1)

        emb = EmbNormalizer()
        total_loss = dict()
        f1score = dict()

        train_inlier_index, train_outlier_index = train_test_split(
            range(len(x_train_np)), train_size= 1 - ratio_anomaly, random_state = 0)
        valid_inlier_index, valid_outlier_index = train_test_split(
            range(len(x_valid_np)), train_size = 1 - ratio_anomaly, random_state = 0)
        
        # creating aug data with fixed length/level
        x_aug_list, labels_list = list(), list()
        for anomaly_type in anomaly_types:
            x_aug, labels = list(), list()
            for i in train_outlier_index:
                x = x_train_np[i]
                # Use fixed level and length
                fixed_level = 0.5
                fixed_length = 0.3
                xa, l = inject(anomaly_type = anomaly_type, ts = x,
                              config = [fixed_level, np.random.uniform(0, 0.5), fixed_length])
                x_aug.append(xa)
                labels.append(l)
            x_aug_list.append(x_aug)
            labels_list.append(labels)

        z_aug = model(torch.cat([torch.tensor(np.array(x_aug)).to(args.device)
                                for x_aug in x_aug_list], dim = 0).float().unsqueeze(1)).detach()
        
        # Initialize the normalizer with normal and fixed augmented data
        z_train_t, z_valid_t, z_aug_t = emb(
            z_train[train_inlier_index].clone().squeeze(),
            z_valid[valid_inlier_index].clone().squeeze(),
            z_aug.clone().squeeze()
        )
        
        def argument(x_np, configs, outlier_index, config_name):
            x_configs_augs_dict, configs_labels_dict = dict(), dict()
            for anomaly_type in anomaly_types:
                x_configs_augs, configs_labels = list(), list()
                for config in configs:
                    x_augs, labels = list(), list()
                    for i in outlier_index:
                        if config_name == 'level':
                            x_aug, label = inject(anomaly_type=anomaly_type, ts=x_np[i],
                                                 config=[config, np.random.uniform(0, 0.5), fixed_length])
                        elif config_name == 'length':
                            x_aug, label = inject(anomaly_type=anomaly_type, ts=x_np[i],
                                                 config=[fixed_level, np.random.uniform(0, 0.5), config])
                        else:
                            raise Exception('Unsupported config')
                        x_augs.append(x_aug)
                        labels.append(label)
                    x_configs_augs.append(x_augs)
                    configs_labels.append(labels)
                x_configs_augs_dict[anomaly_type] = x_configs_augs
                configs_labels_dict[anomaly_type] = configs_labels
            return x_configs_augs_dict, configs_labels_dict

        # Generate augmentations with individual parameter variation
        # train level aug
        x_train_level_aug, train_level_labels = argument(
            x_np=x_train_np, configs=train_levels, outlier_index=train_outlier_index, config_name='level')
        # train length aug
        x_train_length_aug, train_length_labels = argument(
            x_np=x_train_np, configs=train_lengths, outlier_index=train_outlier_index, config_name='length')
        # valid level aug
        x_valid_level_aug, valid_level_labels = argument(
            x_np=x_valid_np, configs=valid_levels, outlier_index=valid_outlier_index, config_name='level')
        # valid length aug
        x_valid_length_aug, valid_length_labels = argument(
            x_np=x_valid_np, configs=valid_lengths, outlier_index=valid_outlier_index, config_name='length')

        # Get embeddings for the augmented data
        z_train_level_aug = {anomaly_type: [
            model(torch.tensor(np.array(level_x_aug)).float().unsqueeze(1).to(args.device)).detach() for
            level_x_aug in x_train_level_aug[anomaly_type]] for anomaly_type in anomaly_types}
        z_train_length_aug = {anomaly_type: [
            model(torch.tensor(np.array(length_x_aug)).float().unsqueeze(1).to(args.device)).detach() for
            length_x_aug in x_train_length_aug[anomaly_type]] for anomaly_type in anomaly_types}
        z_valid_level_aug = {anomaly_type: [
            model(torch.tensor(np.array(level_x_aug)).float().unsqueeze(1).to(args.device)).detach() for
            level_x_aug in x_valid_level_aug[anomaly_type]] for anomaly_type in anomaly_types}
        z_valid_length_aug = {anomaly_type: [
            model(torch.tensor(np.array(length_x_aug)).float().unsqueeze(1).to(args.device)).detach() for
            length_x_aug in x_valid_length_aug[anomaly_type]] for anomaly_type in anomaly_types}

        # Normalize the embeddings
        z_train_level_aug_t = {anomaly_type: [emb.normalize(z_aug) for z_aug in z_train_level_aug[anomaly_type]] for
                              anomaly_type in anomaly_types}
        z_train_length_aug_t = {anomaly_type: [emb.normalize(z_aug) for z_aug in z_train_length_aug[anomaly_type]]
                               for anomaly_type in anomaly_types}
        z_valid_level_aug_t = {anomaly_type: [emb.normalize(z_aug) for z_aug in z_valid_level_aug[anomaly_type]] for
                              anomaly_type in anomaly_types}
        z_valid_length_aug_t = {anomaly_type: [emb.normalize(z_aug) for z_aug in z_valid_length_aug[anomaly_type]]
                               for anomaly_type in anomaly_types}

        # function to create augmentations with both level and length varying
        def mixed_argument(x_np, levels, lengths, outlier_index):
            # Structure: {anomaly_type: {(level, length): [augmented_samples]}}
            x_mixed_augs = defaultdict(lambda: defaultdict(list))
            mixed_labels = defaultdict(lambda: defaultdict(list))
            
            # For each anomaly type
            for anomaly_type in anomaly_types:
                # For each level-length combination
                for level in levels:
                    for length in lengths:
                        # Generate augmentations for this config
                        for i in outlier_index:
                            x = x_np[i]
                            x_aug, label = inject(
                                anomaly_type=anomaly_type, 
                                ts=x,
                                config=[level, np.random.uniform(0, 0.5), length]
                            )
                            x_mixed_augs[anomaly_type][(level, length)].append(x_aug)
                            mixed_labels[anomaly_type][(level, length)].append(label)
            
            return x_mixed_augs, mixed_labels
        
        # Generate mixed augmentations for train and validation
        x_train_mixed_augs, train_mixed_labels = mixed_argument(
            x_np=x_train_np, levels=mixed_levels, lengths=mixed_lengths, outlier_index=train_outlier_index)
        
        x_valid_mixed_augs, valid_mixed_labels = mixed_argument(
            x_np=x_valid_np, levels=mixed_levels, lengths=mixed_lengths, outlier_index=valid_outlier_index)
        
        # Get embeddings for mixed augmentations
        z_train_mixed_augs = {}
        z_valid_mixed_augs = {}
        
        for anomaly_type in anomaly_types:
            z_train_mixed_augs[anomaly_type] = {}
            z_valid_mixed_augs[anomaly_type] = {}
            
            for config in x_train_mixed_augs[anomaly_type].keys():
                # Get embeddings for this config
                z_train_mixed_augs[anomaly_type][config] = model(
                    torch.tensor(np.array(x_train_mixed_augs[anomaly_type][config]))
                    .float().unsqueeze(1).to(args.device)).detach()
            
            for config in x_valid_mixed_augs[anomaly_type].keys():
                # Get embeddings for this config
                z_valid_mixed_augs[anomaly_type][config] = model(
                    torch.tensor(np.array(x_valid_mixed_augs[anomaly_type][config]))
                    .float().unsqueeze(1).to(args.device)).detach()
        
        # Normalize mixed embeddings
        z_train_mixed_augs_t = {}
        z_valid_mixed_augs_t = {}
        
        for anomaly_type in anomaly_types:
            z_train_mixed_augs_t[anomaly_type] = {}
            z_valid_mixed_augs_t[anomaly_type] = {}
            
            for config in z_train_mixed_augs[anomaly_type].keys():
                z_train_mixed_augs_t[anomaly_type][config] = emb.normalize(z_train_mixed_augs[anomaly_type][config])
            
            for config in z_valid_mixed_augs[anomaly_type].keys():
                z_valid_mixed_augs_t[anomaly_type][config] = emb.normalize(z_valid_mixed_augs[anomaly_type][config])
        
        # Calculate Wasserstein distance between train and validation mixed augmentations
        W_loss = geomloss.SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.9)
        
        # Create alignment matrices for each anomaly type
        alignment_matrices = {}
        
        for anomaly_type in anomaly_types:
            # Get all configs for this anomaly type
            train_configs = list(z_train_mixed_augs_t[anomaly_type].keys())
            valid_configs = list(z_valid_mixed_augs_t[anomaly_type].keys())
            
            # Create matrix to store alignment scores
            alignment_matrix = np.zeros((len(train_configs), len(valid_configs)))
            
            # Calculate WD between each pair of train and validation configs
            for i, train_config in enumerate(train_configs):
                for j, valid_config in enumerate(valid_configs):
                    train_emb = z_train_mixed_augs_t[anomaly_type][train_config]
                    valid_emb = z_valid_mixed_augs_t[anomaly_type][valid_config]
                    
                    # Calculate Wasserstein distance
                    wd = W_loss(
                        train_emb.view(train_emb.size(0), -1),
                        valid_emb.view(valid_emb.size(0), -1)
                    ).item()
                    
                    alignment_matrix[i, j] = wd
            
            alignment_matrices[anomaly_type] = {
                'matrix': alignment_matrix,
                'train_configs': train_configs,
                'valid_configs': valid_configs
            }
        
        # Visualize alignment matrices
        for anomaly_type in anomaly_types:
            alignment_matrix = alignment_matrices[anomaly_type]['matrix']
            train_configs = alignment_matrices[anomaly_type]['train_configs']
            valid_configs = alignment_matrices[anomaly_type]['valid_configs']
            
            plt.figure(figsize=(12, 10))
            plt.imshow(alignment_matrix, cmap='viridis', interpolation='nearest')
            plt.colorbar(label='Wasserstein Distance (Lower is Better)')
            plt.title(f'Alignment Matrix for {anomaly_type} Anomalies')
            
            # Create labels for configs
            train_labels = [f"L:{c[0]},l:{c[1]}" for c in train_configs]
            valid_labels = [f"L:{c[0]},l:{c[1]}" for c in valid_configs]
            
            plt.xticks(range(len(valid_configs)), valid_labels, rotation=90)
            plt.yticks(range(len(train_configs)), train_labels)
            plt.xlabel('Validation Configurations')
            plt.ylabel('Training Configurations')
            
            # Highlight diagonal elements (same config in train and validation)
            for i in range(min(len(train_configs), len(valid_configs))):
                if train_configs[i] == valid_configs[i]:
                    plt.plot(i, i, 'ro', markersize=10, markerfacecolor='none')
            
            # Save the plot
            os.makedirs(f'logs/training/{args.trail}', exist_ok=True)
            plt.tight_layout()
            plt.savefig(f'logs/training/{args.trail}/mixed_alignment_{anomaly_type}.pdf')
            plt.close()
        
        # ====== VISUALIZE EMBEDDINGS WITH BOTH ORIGINAL AND MIXED AUGMENTATIONS ======
        # Prepare a visualization of the embedding space using t-SNE
        
        # Combine all embeddings for visualization
        all_embs = [
            z_train_t,  # Normal train
            z_valid_t   # Normal validation
        ]
        
        # Add original augmentations
        for anomaly_type in anomaly_types:
            for emb_list in z_train_level_aug_t[anomaly_type]:
                all_embs.append(emb_list)
            for emb_list in z_valid_level_aug_t[anomaly_type]:
                all_embs.append(emb_list)
            for emb_list in z_train_length_aug_t[anomaly_type]:
                all_embs.append(emb_list)
            for emb_list in z_valid_length_aug_t[anomaly_type]:
                all_embs.append(emb_list)
        
        # Add mixed augmentations
        for anomaly_type in anomaly_types:
            for config in z_train_mixed_augs_t[anomaly_type].keys():
                all_embs.append(z_train_mixed_augs_t[anomaly_type][config])
            for config in z_valid_mixed_augs_t[anomaly_type].keys():
                all_embs.append(z_valid_mixed_augs_t[anomaly_type][config])
        
        # Convert to numpy for t-SNE
        all_embs_np = torch.cat(all_embs, dim=0).cpu().numpy()
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embs_2d = tsne.fit_transform(all_embs_np)
        
        # Plot the embeddings
        plt.figure(figsize=(15, 12))
        
        # Track the current index in the embedding array
        idx = 0
        
        # Plot normal data
        normal_train_count = len(z_train_t)
        normal_valid_count = len(z_valid_t)
        
        plt.scatter(
            embs_2d[idx:idx+normal_train_count, 0],
            embs_2d[idx:idx+normal_train_count, 1],
            c='blue', marker='o', alpha=0.5, label='Train Normal'
        )
        idx += normal_train_count
        
        plt.scatter(
            embs_2d[idx:idx+normal_valid_count, 0],
            embs_2d[idx:idx+normal_valid_count, 1],
            c='green', marker='o', alpha=0.5, label='Validation Normal'
        )
        idx += normal_valid_count
        
        # Define colors for mixed anomalies
        colors = plt.cm.rainbow(np.linspace(0, 1, len(mixed_levels) * len(mixed_lengths)))
        color_idx = 0
        
        # Create mapping of configs to colors
        config_colors = {}
        for level in mixed_levels:
            for length in mixed_lengths:
                config_colors[(level, length)] = colors[color_idx]
                color_idx += 1
        
        # Plot original augmentations (simplified, just to move the index)
        legend_elements = [
            Line2D([0], [0], marker='o', color='blue', label='Train Normal', markersize=10),
            Line2D([0], [0], marker='o', color='green', label='Validation Normal', markersize=10)
        ]
        
        # Skip through original augmentations (we focus on mixed)
        for anomaly_type in anomaly_types:
            for i in range(len(z_train_level_aug_t[anomaly_type])):
                idx += len(z_train_level_aug_t[anomaly_type][i])
            for i in range(len(z_valid_level_aug_t[anomaly_type])):
                idx += len(z_valid_level_aug_t[anomaly_type][i])
            for i in range(len(z_train_length_aug_t[anomaly_type])):
                idx += len(z_train_length_aug_t[anomaly_type][i])
            for i in range(len(z_valid_length_aug_t[anomaly_type])):
                idx += len(z_valid_length_aug_t[anomaly_type][i])
        
        # Plot mixed augmentations
        for anomaly_type in anomaly_types:
            for config in z_train_mixed_augs_t[anomaly_type].keys():
                emb_count = len(z_train_mixed_augs_t[anomaly_type][config])
                color = config_colors[config]
                plt.scatter(
                    embs_2d[idx:idx+emb_count, 0],
                    embs_2d[idx:idx+emb_count, 1],
                    c=[color], marker='*', alpha=0.7,
                    label=f'Train {anomaly_type} L:{config[0]} l:{config[1]}'
                )
                legend_elements.append(
                    Line2D([0], [0], marker='*', color='w',
                           label=f'Train {anomaly_type} L:{config[0]} l:{config[1]}',
                           markerfacecolor=color, markersize=10)
                )
                idx += emb_count
        
        for anomaly_type in anomaly_types:
            for config in z_valid_mixed_augs_t[anomaly_type].keys():
                emb_count = len(z_valid_mixed_augs_t[anomaly_type][config])
                color = config_colors[config]
                plt.scatter(
                    embs_2d[idx:idx+emb_count, 0],
                    embs_2d[idx:idx+emb_count, 1],
                    c=[color], marker='x', alpha=0.7,
                    label=f'Valid {anomaly_type} L:{config[0]} l:{config[1]}'
                )
                legend_elements.append(
                    Line2D([0], [0], marker='x', color='w',
                           label=f'Valid {anomaly_type} L:{config[0]} l:{config[1]}',
                           markerfacecolor=color, markersize=10)
                )
                idx += emb_count
        
        plt.title('t-SNE Visualization of Time Series Embeddings with Mixed Anomalies')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        
        # Save the main plot
        plt.tight_layout()
        plt.savefig(f'logs/training/{args.trail}/mixed_visualization.pdf')
        plt.close()
        
        # Save the legend separately (it might be large)
        plt.figure(figsize=(15, 10))
        plt.axis('off')
        plt.legend(handles=legend_elements, loc='center', ncol=3, fontsize=10)
        plt.tight_layout()
        plt.savefig(f'logs/training/{args.trail}/mixed_legend.pdf')
        plt.close()
        
        best_matches = {}
        
        for anomaly_type in anomaly_types:
            best_matches[anomaly_type] = {}
            alignment_matrix = alignment_matrices[anomaly_type]['matrix']
            train_configs = alignment_matrices[anomaly_type]['train_configs']
            valid_configs = alignment_matrices[anomaly_type]['valid_configs']
            
            # Find best validation config for each training config
            for i, train_config in enumerate(train_configs):
                best_j = np.argmin(alignment_matrix[i, :])
                best_valid_config = valid_configs[best_j]
                
                best_matches[anomaly_type][train_config] = {
                    'best_match': best_valid_config,
                    'distance': alignment_matrix[i, best_j]
                }
        
        # Return original structure plus alignment matrices and best matches
        result = {
            'total_loss': total_loss,
            'f1score': f1score,
            'alignment_matrices': alignment_matrices,
            'best_matches': best_matches
        }
        
        return result