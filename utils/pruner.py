import torch
import sys
sys.path.append("..")
from models import resnet, densenet
import copy
import numpy as np
from utils.utils import Pruner

class ResNetPruner(Pruner):
    def __init__(self, model, block_nums):
        print("Initializing ResNetPruner...")
        self.model = model
        self.block_nums = block_nums
    
    def get_blocks_from_model(self, model):
        blocks_original = []
        for m in model.modules():
            if isinstance(m, resnet.Bottleneck):
                blocks_original.append(m)

        return blocks_original

    def construct_model_new(self, block_nums, solution):
        block_nums_new = []
        for i in range(len(block_nums)):
            begin = sum(block_nums[:i])
            end = sum(block_nums[:i+1])
            solution_stage = solution[begin:end]
            block_nums_new.append(int(sum(solution_stage)))
        # import pdb; pdb.set_trace()
        model_new = resnet.ResNet(resnet.Bottleneck, block_nums_new, num_classes=1000)

        return model_new, block_nums_new

    def get_layers_from_block(self, blocks):
        layers_in_block = []
        for m in blocks.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.BatchNorm2d):
                layers_in_block.append(m)

        return layers_in_block

    def copy_block(self, block_original, block_new):
        layers_original = self.get_layers_from_block(block_original)
        layers_new = self.get_layers_from_block(block_new)

        for l_original, l_new in zip(layers_original,layers_new):
            if isinstance(l_original, torch.nn.Conv2d) and isinstance(l_new, torch.nn.Conv2d):
                # print(l_original.weight.data.shape)
                # print(l_new.weight.data.shape)
                l_new.weight.data = l_original.weight.data

            if isinstance(l_original, torch.nn.BatchNorm2d) and  isinstance(l_new, torch.nn.BatchNorm2d):
                l_new.weight = l_original.weight
                l_new.bias = l_original.bias
                l_new.running_mean = l_original.running_mean
                l_new.running_var = l_original.running_var

    def copy_other_layers(self, model_original, model_new):
        keys = ['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'fc.weight', 'fc.bias']
        for k in keys:
            model_new.state_dict()[k][:] = model_original.state_dict()[k]


    def prune_block(self, model_original, block_nums, solution_compact):
        # construct full solution from compact solution (add two more 1s)
        solution = np.ones(sum(block_nums))
        block_nums_compact = [x-1 for x in block_nums]

        # copy them to prunable positions 
        # first block in the first stage are responsible for dimension reduction, not prunable
        # (first block in the later two stages are used for downsampling, not prunable)
        for i in range(len(block_nums)):
            solution[sum(block_nums[:i])+1:sum(block_nums[:i+1])] = solution_compact[sum(block_nums_compact[:i]):sum(block_nums_compact[:i+1])]

        blocks_original = self.get_blocks_from_model(model_original)
        model_new, block_nums_new = self.construct_model_new(block_nums, solution)
        blocks_new = self.get_blocks_from_model(model_new)
        for idx, flag in enumerate(solution):
            if flag==1:
                self.copy_block(blocks_original[idx], blocks_new[int(sum(solution[:idx]))])
        self.copy_other_layers(model_original, model_new)
        return model_new, block_nums_new
    
    def prune(self, solution):
        return self.prune_block(self.model, self.block_nums, solution)
    
# Densenet Pruner
class DenseNetPruner(Pruner):
    def __init__(self, model, dense_layer_nums, num_init_features=64, growth_rate=32):
        self.model = model
        self.dense_layer_nums = dense_layer_nums
        self.num_init_features = num_init_features
        self.growth_rate = growth_rate

    def get_blocks_from_model(self, model):
            blocks_original = []
            for m in model.modules():
                if isinstance(m, densenet._DenseBlock):
                    blocks_original.append(m)

            return blocks_original

    def get_transitions_from_model(self, model):
            transitions_original = []
            for m in model.modules():
                if isinstance(m, densenet._Transition):
                    transitions_original.append(m)
            return transitions_original

    def get_final_two_layers(self, model):
        bn_final = list(model.modules())[-2] # BN
        fc_final = list(model.modules())[-1] # FC
        return bn_final, fc_final

    def get_first_two_layers(self, model):
        first_conv = list(model.modules())[2]  # First Conv2d
        first_bn = list(model.modules())[3]  # First BatchNorm
        return first_conv, first_bn

    def get_dense_layers_from_block(self, blocks):
        dense_layers_in_block = []
        for m in blocks.modules():
            if isinstance(m, densenet._DenseLayer):
                dense_layers_in_block.append(m)
        return dense_layers_in_block

    def get_layers_from_block(self, blocks):
        layers_in_block = []
        for m in blocks.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.BatchNorm2d):
                layers_in_block.append(m)

        return layers_in_block

    def copy_layer_weights_with_mask(self, layer_original, layer_new, mask_original):
        mask = np.where(mask_original)[0]
        if isinstance(layer_original, torch.nn.Conv2d) and isinstance(layer_new, torch.nn.Conv2d):
            layer_new.weight.data = layer_original.weight[:, mask, ...]

        if isinstance(layer_original, torch.nn.BatchNorm2d) and isinstance(layer_new, torch.nn.BatchNorm2d):
            layer_new.weight.data = layer_original.weight[mask]
            layer_new.bias.data = layer_original.bias[mask]
            layer_new.running_mean.data = layer_original.running_mean[mask]
            layer_new.running_var.data = layer_original.running_var[mask]

        if isinstance(layer_original, torch.nn.Linear) and isinstance(layer_new, torch.nn.Linear):
            layer_new.weight.data = layer_original.weight[:, mask]
            layer_new.bias.data = layer_original.bias

    def copy_layer_weights(self, dense_layers_original, dense_layers_new):
        if isinstance(dense_layers_original, densenet._DenseLayer):
            for i in range(len(dense_layers_original)):
                layer_original = dense_layers_original[i]
                layer_new = dense_layers_new[i]
                if isinstance(layer_original, torch.nn.Conv2d) and isinstance(layer_new, torch.nn.Conv2d):
                    layer_new.weight.data = layer_original.weight.data

                if isinstance(layer_original, torch.nn.BatchNorm2d) and  isinstance(layer_new, torch.nn.BatchNorm2d):
                    layer_new.weight.data = layer_original.weight
                    layer_new.bias.data = layer_original.bias
                    layer_new.running_mean.data = layer_original.running_mean
                    layer_new.running_var.data = layer_original.running_var
        else:
            layer_original = dense_layers_original
            layer_new = dense_layers_new

            if isinstance(layer_original, torch.nn.Linear) and isinstance(layer_new, torch.nn.Linear):
                layer_new.weight.data = layer_original.weight.data
                layer_new.bias.data = layer_original.bias.data

            if isinstance(layer_original, torch.nn.Conv2d) and isinstance(layer_new, torch.nn.Conv2d):
                layer_new.weight.data = layer_original.weight.data

            if isinstance(layer_original, torch.nn.BatchNorm2d) and  isinstance(layer_new, torch.nn.BatchNorm2d):
                layer_new.weight.data = layer_original.weight
                layer_new.bias.data = layer_original.bias
                layer_new.running_mean.data = layer_original.running_mean
                layer_new.running_var.data = layer_original.running_var

    def prune_block(self, dense_layers_original, num_features_input, growth_rate, block_id=0, prune_list = [1,3]):
        dense_layers_original_temp = copy.deepcopy(dense_layers_original)
        dense_layer_masks = []
        for i in range(len(dense_layers_original)+1):
            dense_layer_masks.append(np.ones(num_features_input+growth_rate*i, dtype=np.uint8))

        for idx in prune_list:
            for d_idx in range(idx+1, len(dense_layers_original)+1):
                begin_idx = dense_layer_masks[d_idx].shape[0] - (d_idx - idx) * growth_rate
                end_idx = dense_layer_masks[d_idx].shape[0] - (d_idx - idx - 1) * growth_rate
                dense_layer_masks[d_idx][begin_idx:end_idx] = False

        # Prune the first conv layer in the remaining dense layers   
        for idx in prune_list:
            for d_idx in range(idx+1, len(dense_layers_original)):
                mask_temp = dense_layer_masks[d_idx]
                layers_in_dense_layer = dense_layers_original[d_idx]
                layers_in_dense_layer_temp = dense_layers_original_temp[d_idx]
                self.copy_layer_weights_with_mask(layers_in_dense_layer[0], layers_in_dense_layer_temp[0], mask_temp) # BN
                self.copy_layer_weights_with_mask(layers_in_dense_layer[1], layers_in_dense_layer_temp[1], mask_temp) # ReLU
                self.copy_layer_weights_with_mask(layers_in_dense_layer[2], layers_in_dense_layer_temp[2], mask_temp) # Conv2d 
        return dense_layers_original_temp, dense_layer_masks[-1]

    def prune_transition(self, layers_in_transition_layer_source, layers_in_transition_layer_dest, mask):
        for layer_source, layer_dest in zip(layers_in_transition_layer_source, layers_in_transition_layer_dest):
            self.copy_layer_weights_with_mask(layer_source, layer_dest, mask) 
        
        
    def prune_densenet(self, model, transition_config, solution, num_init_features=64, growth_rate=32):
        # Pruned model
        pruning_solution = {}
        pruning_solution[0] = list(np.where(solution[0:sum(transition_config[:1])]==0)[0])
        pruning_solution[1] = list(np.where(solution[sum(transition_config[:1]):sum(transition_config[:2])]==0)[0])
        pruning_solution[2] = list(np.where(solution[sum(transition_config[:2]):sum(transition_config[:3])]==0)[0])
        pruning_solution[3] = list(np.where(solution[sum(transition_config[:3]):]==0)[0])

        block_config=tuple(num-len(pruning_solution[idx]) for idx, num in enumerate(transition_config))
        
        # Pruned model
        model_new = densenet.DenseNet_with_custom_transition(num_init_features=num_init_features, growth_rate=growth_rate, 
                                                             block_config=block_config,
                                                             transition_config=transition_config)
        # Dense Blocks
        blocks = self.get_blocks_from_model(model)
        blocks_new = self.get_blocks_from_model(model_new)

        # Transition layers
        transitions = self.get_transitions_from_model(model)
        transitions_new = self.get_transitions_from_model(model_new)

        # Last two layers (bn and fc)
        bn_final, fc_final = self.get_final_two_layers(model)
        bn_final_new, fc_final_new = self.get_final_two_layers(model_new)

        # First two layers (Conv7x7 and bn)
        conv_first, bn_first = self.get_first_two_layers(model)
        conv_first_new, bn_first_new = self.get_first_two_layers(model_new)

        ## Pruning
        for block_id, prune_list in pruning_solution.items():
            # print(block_id, prune_list)
            dense_layers_in_block_original = blocks[block_id]
            dense_layers_in_block_pruned, mask_transition = self.prune_block(dense_layers_in_block_original, num_init_features, growth_rate, block_id, prune_list)
            # calculate the number of channels as output of each transition layer (halved for compression)
            num_init_features = num_init_features + transition_config[block_id]*growth_rate
            num_init_features = num_init_features//2
            # Copy to the blocks in the new model
            # Dealing with the first block
            dense_layers_in_block_new = blocks_new[block_id]
            # print(len(dense_layers_in_block_original), len(dense_layers_in_block_pruned), len(dense_layers_in_block_new))
            # Skip pruned layers
            idx_to_copy = [x for x in range(len(dense_layers_in_block_pruned)) if x not in prune_list]
            # Copy pruned layers to the new model
            for i in range(len(dense_layers_in_block_new)):
                # print(i, type(dense_layers_in_block_pruned[idx_to_copy[i]]))
                self.copy_layer_weights(dense_layers_in_block_pruned[idx_to_copy[i]], dense_layers_in_block_new[i])

            # Prune transition layer for the first three blocks
            if block_id<3:
                self.prune_transition(transitions[block_id], transitions_new[block_id], mask_transition)
            else:
                # Copy weights for the final batch norm and fc layers
                self.copy_layer_weights_with_mask(bn_final, bn_final_new, mask_transition)
                self.copy_layer_weights_with_mask(fc_final, fc_final_new, mask_transition)
        
        # Copy weights for the first conv and bn layers
        self.copy_layer_weights(conv_first, conv_first_new)
        self.copy_layer_weights(bn_first, bn_first_new)
        
        return model_new, block_config

    def prune(self, solution):
        return self.prune_densenet(self.model, self.dense_layer_nums, solution, self.num_init_features, self.growth_rate)