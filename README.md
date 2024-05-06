# Orchid Classifier

This repository contains a TensorFlow-based classifier for identifying various species of orchids. The classifier is trained on the Orchids 52 Dataset, a collection of images featuring different orchid species.

**Description:** This model is trained to classify images of orchids into 52 different species. The dataset used for training contains images of various orchid species.

**Model Architecture:** The classifier is built using TensorFlow and Keras. It consists of a convolutional neural network (CNN) with multiple convolutional and pooling layers followed by fully connected layers for classification.

**Dataset:** The model is trained on the Orchids 52 Dataset, which can be accessed [here](https://figshare.com/articles/dataset/Orchids_52_Dataset/12896336/1).

**Dependencies:**
- TensorFlow
- numpy


### Usage

Install dependencies:
```bash
pip install -r requirements.txt
```


### Classes

The model can classify the following orchid species:
- anoectochilus burmanicus rolfe
- bulbophyllum auricomum lindl
- bulbophyllum dayanum rchb
- bulbophyllum lasiochilum par. & rchb
- bulbophyllum limbatum
- bulbophyllum longissimum (ridl.) ridl
- bulbophyllum medusae (lindl.) rchb
- bulbophyllum patens king ex hk.f
- bulbophyllum rufuslabram
- bulbophyllum siamensis rchb
- calenthe rubens
- chiloschista parishii seidenf
- chiloschista viridiflora seidenf
- cymbidium aloifolium (l.) sw
- dendrobium chrysotoxum lindl
- dendrobium farmeri paxt
- dendrobium fimbriatum hook
- dendrobium lindleyi steud
- dendrobium pulchellum roxb
- dendrobium pulchellum
- dendrobium secundum bl-lindl
- dendrobium senile par. & rchb.f
- dendrobium signatum rchb. f
- dendrobium thyrsiflorum rchb. f
- dendrobium tortile lindl
- dendrobium tortile
- hygrochillus parishii var. marrioftiana (rchb.f.)
- paphiopedilum bellatulum
- paphiopedilum callosum
- paphiopedilum charlesworthii
- paphiopedilum concolor
- paphiopedilum exul
- paphiopedilum godefroyae
- paphiopedilum gratrixianum
- paphiopedilum henryanum
- paphiopedilum intanon-villosum
- paphiopedilum niveum (rchb.f.) stein
- paphiopedilum parishii
- paphiopedilum spicerianum
- paphiopedilum sukhakulii
- pelatantheria bicuspidata (rolfe ex downie) tang & wang
- pelatantheria insectiflora (rchb.f.) ridl
- phaius tankervilleae (banks ex i' heritier) blume
- phalaenopsis cornucervi (breda) bl. & rchb.f
- rhynchostylis gigantea (lindl.) ridl
- trichoglottis orchideae (koern) garay
- bulbophyllum auratum Lindl
- bulbophyllum morphologorum Krzl
- dendrobium cumulatum Lindl
- maxiralia tenui folia
- paphiopedilum vejvarutianum O. Gruss & Roellke
- oncidium goldiana

### Links

The dataset used for training the model can be found [here](https://figshare.com/articles/dataset/Orchids_52_Dataset/12896336/1).

The model  can be found [here](https://huggingface.co/jero98772/orchid_clasifier).

