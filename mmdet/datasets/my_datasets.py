from .coco import CocoDataset
from .registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class MyDataset(CocoDataset):
    CLASSES = ('a', 'b', 'c', 'd', 'e')


@DATASETS.register_module
class PikachuDataset(XMLDataset):

    CLASSES = ('pikachu',)

    def __init__(self, **kwargs):
        super(PikachuDataset, self).__init__(**kwargs)


@DATASETS.register_module
class MAFADatasetV3(XMLDataset):

    CLASSES = ('mask', 'face')

    def __init__(self, **kwargs):
        super(MAFADatasetV3, self).__init__(**kwargs)
