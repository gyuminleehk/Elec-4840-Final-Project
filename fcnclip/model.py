"""
Main FLAIR modeling function.
"""

import torch
import torchvision
import numpy as np
import os
from dictionary import definitions
from torch.cuda.amp import autocast
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModel, AutoTokenizer, logging
logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
PATH_PRETRAINED_WEIGHTS = "./flair_pretrained_weights/"
ID_FLAIR_RESNET_V1 = "flair_resnet.pth"
URL_ID_FLAIR_RESNET_V1 = "1l24_2IzwQdnaa034I0zcyDLs_zMujsbR"


# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def wget_gdrive_secure(fileid, input_dir, filename):

    os.system("wget 'https://drive.usercontent.google.com/download?"
              "id=$fileid&"
              "export=download&"
              "authuser=0&"
              "confirm=t&"
              "uuid=40cc00ae-7d0b-4b86-b368-f0a37ebf480c&at=AIrpjvMO67CEnxRuJ6k2pvgHJSxq%3A1736964788644'"
              " -c -O '$filename'".
              replace("$fileid", fileid).replace("$filename", input_dir + filename))


class FLAIRModel(torch.nn.Module):
    def __init__(self, vision_type='resnet_v1', bert_type='emilyalsentzer/Bio_ClinicalBERT', vision_pretrained=True,
                 proj_dim=512, proj_bias=False, logit_scale_init_value=0.07, from_checkpoint=True, weights_path=None,
                 out_path=None, image_size=512, caption="A fundus photograph of [CLS]", projection=True,
                 norm_features=True):
        super().__init__()

        # Set attributes
        self.vision_type = vision_type
        self.bert_type = bert_type
        self.vision_pretrained = vision_pretrained
        self.proj_dim = proj_dim
        self.proj_bias = proj_bias
        self.logit_scale_init_value = logit_scale_init_value
        self.from_checkpoint = from_checkpoint
        self.weights_path = weights_path
        self.out_path = out_path
        self.image_size = image_size
        self.caption = caption
        # Use of projection head and feature normalization on visione encoder
        # (only relevant during transferability stage)
        self.projection = projection
        self.norm_features = norm_features

        # Set vision and text encoder
        self.vision_model = VisionModel(vision_type=self.vision_type, pretrained=self.vision_pretrained,
                                        proj_dim=self.proj_dim, proj_bias=self.proj_bias, projection=self.projection,
                                        norm=self.norm_features)
        self.text_model = TextModel(bert_type=self.bert_type, proj_dim=self.proj_dim, proj_bias=self.proj_bias,
                                    projection=self.projection, norm=self.norm_features)

        # learnable temperature for contrastive loss
        self.logit_scale = torch.nn.Parameter(torch.log(torch.tensor(1/self.logit_scale_init_value)))

        # Load pretrained weights
        if from_checkpoint:
            self.load_from_pretrained(self.weights_path)

        # Set model to device
        self.to(device)

    def load_from_pretrained(self, weights_path=None):
        if weights_path is None:
            import zipfile

            input_dir = PATH_PRETRAINED_WEIGHTS
            pretrained_id = ID_FLAIR_RESNET_V1
            pretrained_url_id = URL_ID_FLAIR_RESNET_V1
            weights_path = input_dir + pretrained_id

            if not os.path.exists(input_dir + pretrained_id):
                if not os.path.exists(input_dir):
                    Path(input_dir).mkdir(parents=True, exist_ok=True)

                # download url link
                wget_gdrive_secure(pretrained_url_id, input_dir, filename="weights.zip")

                # unzip
                zipf = zipfile.ZipFile(input_dir + "weights.zip")
                zipf.extractall(input_dir)
                zipf.close()
                print('\n Download model to:', input_dir + pretrained_id)

        state_dict = torch.load(weights_path, map_location='cpu')
        
        # Remove problematic keys
        state_dict = {k: v for k, v in state_dict.items() 
                    if not k.startswith('text_model.model.embeddings.position_ids')}
        
        # Load with strict=False to ignore remaining mismatches
        msg = self.load_state_dict(state_dict, strict=False)
        print('load model weight from:', weights_path)
        print('Missing keys:', msg.missing_keys)
        print('Unexpected keys:', msg.unexpected_keys)

    def softce_clip_loss(self, logits_per_text, logits_per_img, target_pseudo=None):
            """CLIP-style 대칭적 contrastive loss"""

            batch_size = logits_per_text.size(0)
            labels = torch.arange(batch_size, device=logits_per_text.device)
            

            loss_img = torch.nn.functional.cross_entropy(logits_per_img, labels)
            loss_txt = torch.nn.functional.cross_entropy(logits_per_text, labels)
            
            return (loss_img + loss_txt) / 2

    def ce_loss(self, pred_logit, ref):
        ce_loss = torch.nn.functional.cross_entropy(pred_logit, ref)
        return ce_loss

    def compute_logits(self, img_emb, text_emb):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
        return logits_per_text.t()

    def preprocess_text(self, text):

        # Create text prompt
        prompts = [self.caption.replace("[CLS]", category) for category in text]
        # Create text tokens
        text_tokens = self.text_model.tokenize(prompts)
        input_ids = text_tokens["input_ids"].to(device).to(torch.long)
        attention_mask = text_tokens["attention_mask"].to(device).to(torch.long)

        return input_ids, attention_mask

    def compute_text_embeddings(self, categories, domain_knowledge=False):
        # Obtain text embeddings per class
        text_embeds_dict = {}
        for iKey in range(len(categories)):
            # Replace text prompt with expert knowledge descriptions
            if domain_knowledge and categories[iKey] in list(definitions.keys()):
                descriptions = definitions[categories[iKey]]
                if categories[iKey] not in descriptions:
                    if "myopic maculopathy" not in categories[iKey]:  # class names for MMAC are not informative.
                        descriptions.append(categories[iKey])
            else:
                descriptions = [categories[iKey]]

            # Forwards prompts through text encoder
            with torch.no_grad():
                descriptions = [self.caption.replace("[CLS]", iDescription) for iDescription in descriptions]
                text_token = self.text_model.tokenize(descriptions)
                input_ids = text_token["input_ids"].to(device).to(torch.long)
                attention_mask = text_token["attention_mask"].to(device).to(torch.long)

                text_embeds = self.text_model(input_ids, attention_mask)

            text_embeds_dict[categories[iKey]] = text_embeds.mean(0).unsqueeze(0)

        text_embeds_dict = text_embeds_dict
        text_embeds = torch.concat(list(text_embeds_dict.values()))

        return text_embeds_dict, text_embeds


class VisionModel(torch.nn.Module):
    def __init__(self, vision_type='resnet', pretrained=True, proj_dim=512, proj_bias=False, projection=True,
                 norm=True):
        super().__init__()
        self.proj_dim = proj_dim

        # Assert vision encoders
        if vision_type not in ['resnet_v1', 'resnet_v2', 'efficientnet']:
            print("Vision model should be one of resnet/efficientnet... using resnet.")
            vision_type = "resnet_v1"

        # Set vision encoder architecture and pretrained weights
        if vision_type == "resnet_v1" or vision_type == "resnet_v2":
            # Set pretrained weights from Imagenet and get model
            if vision_type == "resnet_v1":
                weights = 'IMAGENET1K_V1' if pretrained else None
            elif vision_type == "resnet_v2":
                weights = 'IMAGENET1K_V2' if pretrained else None
            else:
                weights = 'IMAGENET1K_V1' if pretrained else None
            print("Pretrained weights: " + str(weights))
            self.model = torchvision.models.resnet50(weights=weights)
            # Set number of extracted features
            self.vision_dim = 2048
            # Replace classifier by Identity layer
            self.model.fc = torch.nn.Identity()
        elif vision_type == "efficientnet":
            weights = 'IMAGENET1K_V1' if pretrained else None
            self.model = torchvision.models.efficientnet_b7(weights=weights)
            self.vision_dim = 2096

        # Set output dimension
        if projection:
            self.out_dim = self.proj_dim
        else:
            self.out_dim = self.vision_dim

        # Set projection head
        self.projection_head_vision = ProjectionLayer(layer=torch.nn.Linear(self.vision_dim, self.proj_dim,
                                                                            bias=proj_bias),
                                                      projection=projection, norm=norm)

    def forward(self, pixel_values):
        # Forwards trough vision encoder
        embed = self.model(pixel_values)

        # Compute projection from vision embedding to multi-modal projection
        embed = self.projection_head_vision(embed)
        return embed


class TextModel(torch.nn.Module):
    def __init__(self, bert_type='emilyalsentzer/Bio_ClinicalBERT', proj_dim=512, proj_bias=False, projection=True,
                 norm=True):
        super().__init__()

        # Set tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_type)
        self.tokenizer.model_max_length = 77

        # Load text encoder from pretrained
        self.model = AutoModel.from_pretrained(bert_type, output_hidden_states=True)
        self.model.embeddings.register_buffer(
            "position_ids",
            torch.arange(self.model.config.max_position_embeddings).expand((1, -1)),
            persistent=False
        )
        # Set projection head
        self.projection_head_text = ProjectionLayer(layer=torch.nn.Linear(768, proj_dim, bias=proj_bias),
                                                    projection=projection, norm=norm)

    def tokenize(self, prompts_list):
        text_tokens = self.tokenizer(prompts_list, truncation=True, padding=True, return_tensors='pt')
        return text_tokens

    def forward(self, input_ids, attention_mask):
        # Forwards trough text encoder
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2],
                                        output['hidden_states'][-1]])
        embed = last_hidden_states.permute(1, 0, 2, 3).mean(2).mean(1)
        
        # Project the embeddings to the shared space
        embed = self.projection_head_text(embed)
        return embed


class ProjectionLayer(torch.nn.Module):
    def __init__(self, layer, projection=True, norm=True):
        super().__init__()

        self.apply_projection = projection
        self.norm_modality = bool(projection * norm)
        self.norm_projection = norm
        self.projection = layer

    def forward(self, x):

        if self.norm_modality:
            x = x / x.norm(dim=-1, keepdim=True)

        if self.apply_projection:
            x = self.projection(x)
            if self.norm_projection:
                x = x / x.norm(dim=-1, keepdim=True)

        return x