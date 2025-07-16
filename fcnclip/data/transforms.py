"""
Methods for image and text loading, pre-processing and generation
for vision-language pretraining. Also, it includes data augmentation
utilities.
"""
import numpy as np
import random
import torch
import copy

from PIL import Image
from torchvision.transforms import Resize
from kornia.augmentation import RandomHorizontalFlip, RandomAffine, ColorJitter
from typing import Tuple, Any
definitions = {"no diabetic retinopathy": ["no diabetic retinopathy", "no microaneurysms"],
                "mild diabetic retinopathy": ["only few microaneurysms"],
                "moderate diabetic retinopathy": ["many exudates near the macula",
                                                  "many haemorrhages near the macula",
                                                  "retinal thickening near the macula",
                                                  "hard exudates",
                                                  "cotton wool spots",
                                                  "few severe haemorrhages"],
                "severe diabetic retinopathy": ["venous beading",
                                                "many severe haemorrhages",
                                                "intraretinal microvascular abnormality"],
                "proliferative diabetic retinopathy": ["preretinal or vitreous haemorrhage",
                                                       "neovascularization"],
                "no referable diabetic macular edema": ["no apparent exudates"],
                "hard exudates": ["small white or yellowish deposits with sharp margins", "bright lesion"],
                "soft exudates": ["pale yellow or white areas with ill-defined edges", "cotton-wool spot",
                                  "small, whitish or grey, cloud-like, linear or serpentine, slightly elevated lesions"
                                  " with fimbriated edges"],
                "microaneurysms": ["small red dots"],
                "haemorrhages": ["dense, dark red, sharply outlined lesion"],
                "non clinically significant diabetic macular edema": ["presence of exudates outside the radius of one"
                                                                      " disc diameter from the macula center",
                                                                      "presence of exudates"],
                "age related macular degeneration": ["many small drusen", "few medium-sized drusen", "large drusen",
                                                     "macular degeneration"],
                "media haze": ["vitreous haze", "pathological opacity", "the obscuration of fundus details by vitreous"
                                                                        " cells and protein exudation"],
                "drusens": ["yellow deposits under the retina", "numerous uniform round yellow-white lesions"],
                "pathologic myopia": ["anomalous disc, macular atrophy and possible tessellation"],
                "branch retinal vein occlusion": ["occlusion of one of the four major branch retinal veins"],
                "tessellation": ["large choroidal vessels at the posterior fundus"],
                "epiretinal membrane": ["greyish semi-translucent avascular membrane"],
                "laser scar": ["round or oval, yellowish-white with variable black pigment centrally",
                               "50 to 200 micron diameter lesions"],
                "no laser scar": ["no laser scar"],
                "macular scar": ["macular scar"],
                "central serous retinopathy": ["subretinal fluid involving the fovea", "leakage"],
                "optic disc cupping": ["optic disc cupping"],
                "central retinal vein occlusion": ["central retinal vein occlusion"],
                "tortuous vessels": ["tortuous vessels"],
                "asteroid hyalosis": ["multiple sparking, yellow-white, and refractile opacities in the vitreous cavity",
                                      "vitreous opacities"],
                "optic disc pallor": ["pale yellow discoloration that can be segmental or generalized on optic disc"],
                "optic disc edema": ["optic disc edema"],
                "shunt": ["collateral vessels connecting the choroidal and the retinal vasculature",
                          "collateral vessels of large caliber and lack of leakage"],
                "anterior ischemic optic neuropathy": ["anterior ischemic optic neuropathy"],
                "parafoveal telangiectasia": ["parafoveal telangiectasia"],
                "retinal traction": ["retinal traction"],
                "retinitis": ["retinitis"],
                "chorioretinitis": ["chorioretinitis"],
                "exudates": ["small white or yellowish white deposits with sharp margins", "bright lesion"],
                "retinal pigment epithelium changes": ["retinal pigment epithelium changes"],
                "macular hole": ["lesion in the macula", "grayish fovea"],
                "retinitis pigmentosa": ["pigment deposits are present in the periphery"],
                "cotton wool spots": ["cotton wool spots", "soft exudates"],
                "colobomas": ["colobomas"],
                "optic disc pit maculopathy": ["optic disc pit maculopathy"],
                "preretinal haemorrhage": ["preretinal haemorrhage"],
                "myelinated nerve fibers": ["myelinated nerve fibers"],
                "haemorrhagic retinopathy": ["haemorrhagic retinopathy"],
                "central retinal artery occlusion": ["central retinal artery occlusion"],
                "tilted disc": ["tilted disc"],
                "cystoid macular edema": ["cysts in the macula region"],
                "post traumatic choroidal rupture": ["post traumatic choroidal rupture"],
                "choroidal folds": ["choroidal folds"],
                "vitreous haemorrhage": ["vitreous haemorrhage"],
                "macroaneurysm": ["macroaneurysm"],
                "vasculitis": ["vasculitis"],
                "branch retinal artery occlusion": ["branch retinal artery occlusion"],
                "plaque": ["plaque"],
                "haemorrhagic pigment epithelial detachment": ["haemorrhagic pigment epithelial detachment"],
                "collaterals": ["collaterals"],
                "normal": ["healthy", "no findings", "no lesion signs", "no glaucoma", "no retinopathy"],
                "large optic cup": ["abnormality in optic cup"],
                "retina detachment": ["retina detachment"],
                "Vogt-Koyanagi syndrome": ["Vogt-Koyanagi syndrome"],
                "maculopathy": ["maculopathy"],
                "glaucoma": ["optic nerve abnormalities", "abnormal size of the optic cup",
                             "anomalous size in the optic disc"],
                "optic atrophy": ["optic atrophy"],
                "severe hypertensive retinopathy": ["flame shaped hemorrhages at the disc margin, blurred disc margins,"
                                                    " congested retinal veins, papilledema, and secondary macular "
                                                    "exudates", "arterio-venous crossing changes, macular star and "
                                                                "cotton wool spots"],
                "disc swelling and elevation": ["disc swelling and elevation"],
                "dragged disk": ["dragged disk"],
                "congenital disk abnormality": ["disk abnormality", "optic disk lesion"],
                "Bietti crystalline dystrophy": ["Bietti crystalline dystrophy"],
                "peripheral retinal degeneration and break": ["peripheral retinal degeneration and break"],
                "neoplasm": ["neoplasm"],
                "yellow-white spots flecks": ["yellow-white spots flecks"],
                "fibrosis": ["fibrosis"],
                "silicon oil": ["silicon oil"],
                "no proliferative diabetic retinopathy": ["diabetic retinopathy with no neovascularization",
                                                          "no neovascularization"],
                "no glaucoma": ["no glaucoma"],
                "cataract": ["opacity in the macular area"],
                "hypertensive retinopathy": ["possible signs of haemorraghe with blot, dot, or flame-shaped",
                                             "possible presence of microaneurysm, cotton-wool spot, or hard exudate",
                                             "arteriolar narrowing", "vascular wall changes", "optic disk edema"],
                "neovascular age related macular degeneration": ["neovascular age-related macular degeneration"],
                "geographical age related macular degeneration": ["geographical age-related macular degeneration"],
                "acute central serous retinopathy": ["acute central serous retinopathy"],
                "chronic central serous retinopathy": ["chronic central serous retinopathy"],
                "no cataract": ["no cataract signs", "no obscure opacities"],
                "abnormal optic disc": ["abnormal optic disc"],
                "abnormal vessels": ["abnormal vessels"],
                "abnormal macula": ["abnormal macula"],
                "macular edema": ["macular edema"],
                "scar": ["scar"],
                "nevus": ["darkly pigmented lesion found in the back of the eye"],
                "increased cup disc": ["increased cup disc"],
                "intraretinal microvascular abnormalities": ["shunt vessels and appear as abnormal branching or"
                                                             " dilation of existing blood vessels (capillaries) "
                                                             "within the retina", "deeper in the retina than"
                                                             " neovascularization, has blurrier edges, is more"
                                                             " of a burgundy than a red, does not appear on the "
                                                             "optic disc", "vascular loops confined within the"
                                                             " retina"],
                "red small dots": ["microaneurysms"],
                "neovascularisation": ["neovascularisation"],
                "a disease": ["no healthy", "lesions"],
                "superficial haemorrhages": ["superficial haemorrhages"],
                "deep haemorrhages": ["deep haemorrhages"],
                "ungradable": ["no fundus", "very noisy", "noisy"],
                "noisy": ["noisy"],
                "normal macula": ["normal macula"],
                "macular degeneration": ["macular degeneration"],
                "diabetic retinopathy": ["diabetic retinopathy"],
                "no hypertensive retinopathy": ["no presence of hypertensive retinopathy"],
                "mild hypertensive retinopathy": ["mild arteriovenous ratio", "mild tortuosity",
                                                  "focal arteriolar narrowing",
                                                  "arteriovenous nicking"],
                "moderate hypertensive retinopathy": ["moderate arteriovenous ratio", "moderate tortuosity",
                                                      "cotton wool spots",
                                                      "flame-shaped haemorrhages"],
                "malignant hypertensive retinopathy": ["severe arteriovenous ratio", "severe tortuosity",
                                                       "swelling optical disk",
                                                       "flame-shaped haemorrhages"],
                "myopic maculopathy grade cero": ["healthy macula"],
                "myopic maculopathy grade one": ["tessellated fundus"],
                "myopic maculopathy grade two": ["diffuse chorioretinal atrophy"],
                "myopic maculopathy grade three": ["patchy chorioretinal atrophy"],
                "myopic maculopathy grade four": ["macular atrophy"],
            }
# Augmentations for pretraining
augmentations_pretraining = torch.nn.Sequential(RandomHorizontalFlip(p=0.5),
                                                RandomAffine(p=0.25, degrees=(-5, 5), scale=(0.9, 1)),
                                                ColorJitter(p=0.25, brightness=0.2, contrast=0.2))


class AugmentationsSegmentation(torch.nn.Module):
    def __init__(self):
        super(AugmentationsSegmentation, self).__init__()

        # we define and cache our operators as class members
        self.k1 = kornia.augmentation.ColorJitter(p=0.25, brightness=0.2, contrast=0.2)
        self.k2 = kornia.augmentation.RandomHorizontalFlip(p=0.5)
        self.k3 = kornia.augmentation.RandomAffine(p=0.25, degrees=(-5, 5), scale=(0.9, 1))
        self.k4 = kornia.augmentation.RandomCrop(size=(1024, 1024))

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[Any, Any]:
        img_out = img
        # 1. apply color only in image
        # img_out = self.k1(img_out)
        # 2. apply geometric tranform
        #img_out = self.k4(self.k3(self.k2(img_out)))
        img_out = self.k3(self.k2(img_out))

        # 3. infer geometry params to mask
        # TODO: this will change in future so that no need to infer params
        #mask_out = self.k4(self.k3(self.k2(mask, self.k2._params), self.k3._params), self.k4._params)
        mask_out = self.k3(self.k2(mask, self.k2._params), self.k3._params)

        return img_out, mask_out



class LoadImage():
    def __init__(self, target="image_path"):
        self.target = target
        """
        Load, organize channels, and standardize intensity of images.
        """

    def __call__(self, data):
        # Read image
        img = np.array(Image.open(data[self.target]).convert('RGB'), dtype=float)
        if np.max(img) > 1:
            img /= 255

        # channel first
        if len(img.shape) > 2:
            img = np.transpose(img, (2, 0, 1))
        else:
            img = np.expand_dims(img, 0)

        if img.shape[0] > 3:
            img = img[1:, :, :]
        if "image" in self.target:
            if img.shape[0] < 3:
                img = np.repeat(img, 3, axis=0)

        data[self.target.replace("_path", "")] = img
        return data


class ImageScaling():

    """
    Method for image scaling. It includes two options: scaling from canvas, to avoid image distortions,
    and regular scaling trough resizing.
    """

    def __init__(self, size=(512, 512), canvas=True, target="image"):
        self.size = size
        self.canvas = canvas
        self.target = target

        self.transforms = torch.nn.Sequential(
            Resize(self.size),
        )

    def __call__(self, data):
        img = torch.tensor(data[self.target])
        if not self.canvas or (img.shape[-1] == img.shape[-2]):
            img = self.transforms(img)
        else:
            sizes = img.shape[-2:]
            max_size = max(sizes)
            scale = max_size/self.size[0]
            img = Resize((int(img.shape[-2]/scale), int((img.shape[-1]/scale))))(img)
            img = torch.nn.functional.pad(img, (0, self.size[0] - img.shape[-1], 0, self.size[1] - img.shape[-2], 0, 0))

        data[self.target] = img
        return data


class ProduceDescription():

    """
    Method that creates naive text prompts combining a prompt template, atributes (e.g. noisy), and categories
    (e.g. cataract). Also, this method is used to integrate text data with the modality prompt template.
    """

    def __init__(self, caption):
        self.caption = caption

    def __call__(self, data):

        # Create text
        atr_sample = random.sample(data['atributes'], 1)[0] if len(data['atributes']) > 0 else ""
        cat_sample = random.sample(data['categories'], 1)[0] if len(data['categories']) > 0 else ""

        data["sel_category"] = cat_sample
        data["report"] = [self.caption.replace("[ATR]",  atr_sample).replace("[CLS]",  cat_sample).replace("  ", " ")]

        return data


class AugmentDescription():

    """
    Method that augments naive text prompts into expert knowledge prompts by changing the category name
    by expert descriptions of the target category.
    """

    def __init__(self, augment=False):
        self.augment = augment

    def __call__(self, data):

        if self.augment:
            if data["image_name"].split("/")[0] not in ["06_EYENET", "11_STARE", "08_ODIR-5K", "31_JICHI"]:
                if data["sel_category"] in list(definitions.keys()):
                    prompts = [data["sel_category"]] + definitions[data["sel_category"]]
                    new_cat = random.sample(prompts, 1)[0]
                    data["report"][0] = data["report"][0].replace(data["sel_category"], new_cat)
                    data["augmented_category"] = new_cat

        return data


class CopyDict():
    def __call__(self, data):
        d = copy.deepcopy(data)
        return d


class SelectRelevantKeys():

    def __init__(self, target_keys=None):
        if target_keys is None:
            target_keys = ['image', 'report', 'sel_category']
        self.target_keys = target_keys

    def __call__(self, data):
        d = {key: data[key] for key in self.target_keys}
        return d