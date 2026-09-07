from google.cloud import documentai


class DocumentAIPageOCR:
    def __init__(self, processor_name: str) -> None:
        self.processor_name = processor_name
        self.client = documentai.DocumentProcessorServiceAsyncClient()

    async def extract(self, pdf_page: bytes) -> str:
        result = await self.client.process_document(
            request=documentai.ProcessRequest(
                name=self.processor_name,
                raw_document=documentai.RawDocument(content=pdf_page, mime_type="application/pdf"),
            )
        )
        return result.document.text
