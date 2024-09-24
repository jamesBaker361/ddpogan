from transformers.models.clip.modeling_clip import CLIPVisionTransformer,CLIPVisionConfig,CLIPModel
from transformers.models.clip.image_processing_clip import CLIPImageProcessor
from torch import nn,Tensor
from PIL import Image
from typing import List, Union

def weights_init(m):
    classname = m.__class__.__name__
    invalid=set()
    try:
        m.weight.data.normal_(0.0, 0.02)
    except:
        invalid.add(classname)
    #print("couldnt init weights for ",",".join([i for i in invalid]))
    

class ClipDiscriminator(nn.Module):
    def __init__(self, random_init:bool=False,device:str="cpu" ) -> None:
        super().__init__()
        model=CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.vision_model = model.vision_model
        self.vision_model=self.vision_model.to(device)
        self.device=device
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        if random_init:
            self.vision_model.apply(weights_init)
        
        self.classification_head=nn.Sequential(
            nn.Linear(model.vision_embed_dim,256),
            nn.Linear(256,1))
        self.classification_head=self.classification_head.to(device)
        print("model.vision_embed_dim",model.vision_embed_dim )

    def forward(self,images: Union[Image.Image, List[Image.Image]])->Tensor:
        if isinstance(images, list) is False:
            images=[images]
        inputs = self.processor(images=images, return_tensors="pt")
        inputs["pixel_values"]=inputs["pixel_values"].to(self.device)
        outputs = self.vision_model(**inputs)
        pooled_output = outputs.pooler_output
        classification=self.classification_head(pooled_output)
        classification=nn.functional.sigmoid(classification)
        return classification

        


if __name__=='__main__':
    disc=ClipDiscriminator(True)
    img=Image.open("jinx.jpg")
    print(disc(img))