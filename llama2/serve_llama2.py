import requests
from starlette.requests import Request
from typing import Dict
from transformers import pipeline
import torch
import ray
from ray import serve
from fastapi import FastAPI
import os


mytoken = os.getenv("HF_API_TOKEN", None)
address = os.getenv("RAY_CLIENT_URI", None)


#install additionall libraries that will be required for model serving
runtime_env = {"pip": ["transformers", "datasets", 
                       "evaluate", "pyarrow<7.0.0", "accelerate"]}

ray.shutdown()
ray.init(address=f"ray://{address}:10001", runtime_env=runtime_env)
print("Ray cluster is up and running: ", ray.is_initialized())

# 1: Wrap the pretrained  LLAMA2 instruction model in a Serve deployment.
@serve.deployment(num_replicas=1, ray_actor_options={"num_gpus":1, "num_cpus":4})
class RayServeDeployment:
    def __init__(self):
        self._model = pipeline("text-generation", model="google/flan-t5-large", 
                               torch_dtype= torch.float16, device_map="auto", token=mytoken,
                              num_return_sequences=1)

    def __call__(self, request: Request) -> Dict:
        response = self._model(request.query_params["text"])[0]["generated_text"]
        return response

# 2: Deploy the deployment.
serve.run(RayServeDeployment.bind(), host="0.0.0.0")
print("Model is Served!")