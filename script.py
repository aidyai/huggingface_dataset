# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dataset class for Shoe40k dataset."""

import datasets
from datasets.tasks import ImageClassification
import pandas as pd
import json
import requests


_HOMEPAGE = "https://huggingface.co/datasets/aidystark/shoe41k"

_DESCRIPTION = (
"----------------------------------------"    
)

_CITATION = """\

"""

_LICENSE = """\
LICENSE AGREEMENT
=================
"""

_NAMES      = ['Dressing Shoe', 'Boot', 'Crocs', 'Heels', 'Sandals', 'Sneakers']
_CSV   =  "https://huggingface.co/datasets/aidystark/shoe41k/resolve/main/FOOT40K.csv"
_URL   =  "https://huggingface.co/datasets/aidystark/shoe41k/resolve/main/shoe40k"



df = pd.read_csv(_CSV)
imgLabels = df['Label']


class shoe40k(datasets.GeneratorBasedBuilder):
    """-------"""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(names=_NAMES),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            task_templates=[ImageClassification(image_column="image", label_column="label")],
        )

    def _split_generators(self, dl_manager):
        path        = dl_manager.download(_URL)
        image_iters = dl_manager.iter_archive(path)
        return [datasets.SplitGenerator(datasets.Split.TRAIN,gen_kwargs={"images":image_iters,})]
    

    def _generate_examples(self, images):
        """Generate images and labels for splits."""
        idx = 0
        #Iterate through images
        for filepath,image in images:
            yield idx, {
                "image":{"path":filepath, "bytes":image.read()},
                "label":imgLabels[idx]
            }
            idx += 1
        
