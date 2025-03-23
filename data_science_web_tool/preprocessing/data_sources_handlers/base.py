from django.db.models.fields.files import FieldFile


class DataSourceHandlerBase:
    """
    Base class that forces an interface for different data source handlers.
    """

    DEFAULT_ENCODING = "utf-8"

    def __init__(
        self,
        file,
        encoding: str = "utf-8",
    ):
        self.file: FieldFile = file
        self.encoding = encoding

    def load_data(self):
        raise NotImplementedError
