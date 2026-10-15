import pytest


@pytest.fixture
def advisory_record_factory():
    def create_record(
        document_id="GHSA-test-0000-0001",
        package_name="example-package",
        summary="Example vulnerability",
        retrieval_text=None,
    ):
        text = retrieval_text or (
            f"Advisory ID: {document_id}\n"
            f"Affected packages: {package_name}\n"
            f"Summary: {summary}"
        )
        return {
            "document_id": document_id,
            "chunk_id": f"{document_id}_chunk_0",
            "packages": [{"name": package_name}],
            "summary": summary,
            "retrieval_text": text,
        }

    return create_record
