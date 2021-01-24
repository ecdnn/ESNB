from utils.utils import *
from utils.ec import *
from utils.pruner import *
from scipy.stats import bernoulli
import os
       
class IndividualPruning(Individual):
    def __init__(self, args_list, apply_pk=False):
        gene_length = args_list['gene_length']
        self.nn_pruner = args_list.get('nn_pruner')
        self.pk_list = args_list.get('pk')
        
        self.dec = np.zeros(gene_length, dtype=np.uint8)  ## binary
        for i in range(gene_length):
            self.dec[i] = 1  # always begin with 1
            if apply_pk:
                self.dec[i] = bernoulli.rvs(size=1,p=self.pk_list[i])[0] # bernoulli sampling
        self.obj = [0, 0]  # initial obj value will be replaced by evaluate()
        
        #if not args_list.get('init'): 
        #    self.evaluate()
        
    def evaluate(self):
        top1_avg, top5_avg, block_nums_pruned = self.nn_pruner.test(self.dec)
        print("    top1:{}, top5:{}, block_nums_pruned:{}({})".format(top1_avg, top5_avg, block_nums_pruned, sum(block_nums_pruned)))
        self.obj[0], self.obj[1] = (100-top1_avg), np.sum(self.dec)

def initialize_nn_pruner():
    class Arg:
        pass

    args_nn = Arg()

    args_nn.data = "/root/home/dataset/imagenet/"
    args_nn.model_path = "/root/home/zy/code/Imagenet/resnet152/model_best_no_module.pth.tar"
    args_nn.sampled_imgs = "sampled_images.pkl"   # "/root/home/zy/code/Imagenet/sampled_imgs1.pkl"
    args_nn.load_model = True
    args_nn.arch = 'resnet152'
    args_nn.num_blocks = [3, 8, 36, 3]

    args_nn.workers = 0
    args_nn.print_freq = 10

    args_nn.batch_size = 32

    args_nn.gpu = 0
    
    args_nn.pruner = ResNetPruner

    nn_pruner = NetworkPruning(args_nn)
    dec_dim = sum(args_nn.num_blocks)-4 # skip the first block in each stage (dimension reduction and down sampling)
    return nn_pruner, dec_dim

if __name__ == '__main__':
    # configuration
    population = []
    pop_size = 30  # Population size
    n_obj = 2       # Objective variable dimensionality
    
    # Initialize nn pruner
    nn_pruner, dec_dim = initialize_nn_pruner()
    # dec_dim = sum(nn_pruner.args.num_blocks)-4  # Decision variable dimensionality

    gen = 200          # Iteration number

    # EC begin
    g_begin = 0
    
    p_crossover = 1     # crossover probability
    p_mutation = 0.1      # mutation probability
    
    path_save = "results_prune_resnet152/"
    if not os.path.exists(path_save):
        os.mkdir(path_save)

    # Initialize ec
    args_ec = {}
    args_ec['individual'] = IndividualPruning
    args_ec['gene_length'] = dec_dim
    args_ec['nn_pruner'] = nn_pruner 
    
    args_ec['pk'] = compute_pk(nn_pruner)
    
    print("Population initialization...")
    
    population = initialization(pop_size=pop_size, args_list=args_ec)

    if g_begin != 0:
        print("@_@ Loading population from generation {}".format(g_begin))
        population = load_population(population, path_save + "population-{}.pkl".format(g_begin))
    for g in range(g_begin+1, gen):
        begin = time.time()
        
        # Variation
        offspring = variation(population, p_crossover, p_mutation)

        # P+Q
        population.extend(offspring)

        # Environmental Selection
        population = environmental_selection(population, pop_size)

        # Plot
        print('Gen:', g)
        objs1 = [p.obj[0] for p in population]
        objs2 = [p.obj[1] for p in population]

        # Save to file
        plt.figure()
        plt.plot(objs1, objs2, 'bo')
        plt.savefig(path_save + 'result{}.png'.format(g))
        plt.close('all')
        
        time_cost = time.time() - begin
        print("#######   time cost:", time_cost, " s")
        save_population(population, path_save + "population-{}.pkl".format(g))
