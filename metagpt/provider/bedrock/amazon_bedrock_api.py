
import json
from typing import Coroutine, Literal
from metagpt.const import USE_CONFIG_TIMEOUT
from metagpt.provider.llm_provider_registry import register_provider
from metagpt.configs.llm_config import LLMConfig, LLMType
from metagpt.provider.base_llm import BaseLLM
from metagpt.logs import log_llm_stream, logger
from botocore.config import Config
import boto3

from metagpt.provider.bedrock.bedrock_provider import get_provider


@register_provider([LLMType.AMAZON_BEDROCK])
class AmazonBedrockLLM(BaseLLM):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.__client = self.__init_client("bedrock-runtime")
        self.provider = get_provider(self.config.model)

    def __init_client(self, service_name: Literal["bedrock-runtime", "bedrock"]):
        # access key from https://us-east-1.console.aws.amazon.com/iam
        self.__credentital_kwards = {
            "aws_secret_access_key": self.config.secret_key,
            "aws_access_key_id": self.config.access_key,
            "region_name": self.config.region_name
        }
        session = boto3.Session(**self.__credentital_kwards)
        client = session.client(service_name)
        return client

    def list_models(self):
        client = self.__init_client("bedrock")
        # only output text-generation models
        response = client.list_foundation_models(byOutputModality='TEXT')
        summaries = [f'{summary.get("modelId", ""):50} Support Streaming:{summary.get("responseStreamingSupported","")}'
                     for summary in response.get("modelSummaries", {})]
        logger.info("\n"+"\n".join(summaries))

    @property
    def _generate_kwargs(self):
        return {
            "temperature": self.config.get("temperature", 0.3),
        }

    def completion(self, messages: list[dict]):
        request_body = self.provider.get_request_body(
            messages, **self._generate_kwargs)
        response = self.__client.invoke_model(
            modelId=self.config.model, body=request_body
        )
        completions = self.provider.get_choice_text(response)
        return completions

    def _chat_completion_stream(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT):
        request_body = self.provider.get_request_body(
            messages, **self._generate_kwargs)
        response = self.__client.invoke_model_with_response_stream(
            modelId=self.config.model, body=request_body
        )
        collected_content = []

        for event in response.get("body"):
            chunk_text = json.loads(event["chunk"]["bytes"])[
                "outputs"][0]["text"]
            collected_content.append(chunk_text)
            log_llm_stream(chunk_text)

        log_llm_stream("\n")
        full_text = ("".join(collected_content)).lstrip()
        return full_text

    async def acompletion(self, messages: list[dict]):
        return self._achat_completion(messages)

    async def _achat_completion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT):
        # TODO:make it async
        return self.completion(messages)

    async def _achat_completion_stream(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT):
        return self._chat_completion_stream(messages)


if __name__ == '__main__':
    from .config import my_config
    prompt = "write an essay for living on mars in 1000 word"
    messages = [{"role": "user", "content": prompt}]
    llm = AmazonBedrockLLM(my_config)
    llm._chat_completion_stream(messages)
