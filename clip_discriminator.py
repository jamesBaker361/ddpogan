from transformers.models.clip.modeling_clip import CLIPVisionTransformer,CLIPVisionConfig,CLIPModel,CLIPVisionModelWithProjection
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
        self.vision_model=CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
        self.vision_model = self.vision_model.to(device)
        self.device=device
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        if random_init:
            self.vision_model.apply(weights_init)
        
        self.classification_head=nn.Sequential(
            nn.Linear(self.vision_model.hidden_size,512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Linear(512,256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Linear(256,128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128,64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64,1))
        self.classification_head=self.classification_head.to(device)
        print("model.hidden_size",self.vision_model.hidden_size )

    def forward(self,images: Union[Image.Image, List[Image.Image]])->Tensor:
        if isinstance(images, list) is False:
            images=[images]
        inputs = self.processor(images=images, return_tensors="pt")
        inputs["pixel_values"]=inputs["pixel_values"].to(self.device)
        outputs = self.vision_model(**inputs)
        print(outputs)
        image_embeds = outputs.image_embeds
        classification=self.classification_head(image_embeds)
        classification=nn.functional.sigmoid(classification)
        return classification

        


if __name__=='__main__':
    disc=ClipDiscriminator(True)
    img=Image.open("jinx.jpg")
    print(disc(img))