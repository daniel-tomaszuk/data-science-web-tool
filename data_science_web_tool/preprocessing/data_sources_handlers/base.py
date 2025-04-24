from django.db.models.fields.files import FieldFile


class DataSourceHandlerBase:
    """
    Base class that forces an interface for different data source handlers.
    """

    DEFAULT_ENCODING = "utf-8"

    def __init__(
        self,
        file: FieldFile | None = None,
        encoding: str = "utf-8",
        **kwargs,
    ):
        self.file: FieldFile = file
        self.encoding = encoding
        self.kwargs = kwargs

    def load_data(self):
        raise NotImplementedError
