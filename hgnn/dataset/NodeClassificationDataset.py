import torch as th
from dgl.data.utils import load_graphs
from . import BaseDataset, register_dataset


@register_dataset('node_classification')
class NodeClassificationDataset(BaseDataset):
    r"""
    The class *NodeClassificationDataset* is a base class for datasets which can be used in task *node classification*.
    So its subclass should contain attributes such as graph, category, num_classes and so on.
    Besides, it should implement the functions *get_labels()* and *get_split()*.

    Attributes
    -------------
    g : dgl.DGLHeteroGraph
        The heterogeneous graph.
    category : str
        The category(or target) node type need to be predict. In general, we predict only one node type.
    num_classes : int
        The target node  will be classified into num_classes categories.
    has_feature : bool
        Whether the dataset has feature. Default ``False``.
    multi_label : bool
        Whether the node has multi label. Default ``False``. For now, only HGBn-IMDB has multi-label.
    """

    def __init__(self, *args, **kwargs):
        super(NodeClassificationDataset, self).__init__(*args, **kwargs)
        self.g = None
        self.category = None
        self.num_classes = None
        self.has_feature = False
        self.multi_label = False
        self.meta_paths_dict =None
        # self.in_dim = None

    def get_labels(self):
        r"""
        The subclass of dataset should overwrite the function. We can get labels of target nodes through it.

        Notes
        ------
        In general, the labels are th.LongTensor.
        But for multi-label dataset, they should be th.FloatTensor. Or it will raise
        RuntimeError: Expected object of scalar type Long but got scalar type Float for argument #2 target' in call to _thnn_nll_loss_forward
        
        return
        -------
        labels : torch.Tensor
        """
        if 'labels' in self.g.nodes[self.category].data:
            labels = self.g.nodes[self.category].data.pop('labels').long()
        elif 'label' in self.g.nodes[self.category].data:
            labels = self.g.nodes[self.category].data.pop('label').long()
        else:
            raise ValueError('Labels of nodes are not in the hg.nodes[category].data.')
        labels = labels.float() if self.multi_label else labels
        return labels

    def get_split(self, validation=True):
        r"""
        
        Parameters
        ----------
        validation : bool
            Whether to split dataset. Default ``True``. If it is False, val_idx will be same with train_idx.

        We can get idx of train, validation and test through it.

        return
        -------
        train_idx, val_idx, test_idx : torch.Tensor, torch.Tensor, torch.Tensor
        """
        if 'train_mask' not in self.g.nodes[self.category].data:
            self.logger.dataset_info("The dataset has no train mask. "
                  "So split the category nodes randomly. And the ratio of train/test is 8:2.")
            num_nodes = self.g.number_of_nodes(self.category)
            n_test = int(num_nodes * 0.2)
            n_train = num_nodes - n_test
    
            train, test = th.utils.data.random_split(range(num_nodes), [n_train, n_test])
            train_idx = th.tensor(train.indices)
            test_idx = th.tensor(test.indices)
            if validation:
                self.logger.dataset_info("Split train into train/valid with the ratio of 8:2 ")
                random_int = th.randperm(len(train_idx))
                valid_idx = train_idx[random_int[:len(train_idx) // 5]]
                train_idx = train_idx[random_int[len(train_idx) // 5:]]
            else:
                self.logger.dataset_info("Set valid set with train set.")
                valid_idx = train_idx
                train_idx = train_idx
        else:
            train_mask = self.g.nodes[self.category].data.pop('train_mask')
            test_mask = self.g.nodes[self.category].data.pop('test_mask')
            train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
            test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
            if validation:
                if 'val_mask' in self.g.nodes[self.category].data:
                    val_mask = self.g.nodes[self.category].data.pop('val_mask')
                    valid_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
                elif 'valid_mask' in self.g.nodes[self.category].data:
                    val_mask = self.g.nodes[self.category].data.pop('valid_mask').squeeze()
                    valid_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
                else:
                    # RDF_NodeClassification has train_mask, no val_mask
                    self.logger.dataset_info("Split train into train/valid with the ratio of 8:2 ")
                    random_int = th.randperm(len(train_idx))
                    valid_idx = train_idx[random_int[:len(train_idx) // 5]]
                    train_idx = train_idx[random_int[len(train_idx) // 5:]]
            else:
                self.logger.dataset_info("Set valid set with train set.")
                valid_idx = train_idx
                train_idx = train_idx
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx
        # Here set test_idx as attribute of dataset to save results of HGB
        return self.train_idx, self.valid_idx, self.test_idx


@register_dataset('rdf_node_classification')
class RDF_NodeClassification(NodeClassificationDataset):
    r"""
    The RDF dataset will be used in task *entity classification*.
    Dataset Name : aifb/ mutag/ bgs/ am.
    We download from dgl and process it, refer to
    `RDF datasets <https://docs.dgl.ai/api/python/dgl.data.html#rdf-datasets>`_.

    Notes
    ------
    They are all have no feature.
    """

    def __init__(self, dataset_name, *args, **kwargs):
        super(RDF_NodeClassification, self).__init__(*args, **kwargs)
        self.g, self.category, self.num_classes = self.load_RDF_dgl(dataset_name)
        self.has_feature = False

    def load_RDF_dgl(self, dataset):
        if dataset == 'fma':
            data_path = './hgnn/dataset/fma.bin'
            g, _ = load_graphs(data_path)
            kg = g[0].long()
            category = 'track'
            num_classes = 4
        else:
            raise ValueError()

        return kg, category, num_classes