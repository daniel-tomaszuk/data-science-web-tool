import contextlib
import csv

import chardet
from preprocessing.data_sources_handlers.base import DataSourceHandlerBase


class CsvDataSourceHandler(DataSourceHandlerBase):
    SNIFFING_CHARS_COUNT = 2048

    def load_data(self):
        """Reads the uploaded CSV file, detects encoding & dialect, and saves content as JSON."""
        self.file.seek(0)
        raw_data: bytes = self.file.read(self.SNIFFING_CHARS_COUNT)
        detected_encoding: str = (
            chardet.detect(raw_data).get("encoding") or self.DEFAULT_ENCODING
        )
        self.file.seek(0)

        decoded_file = self.file.read().decode(detected_encoding).splitlines()

        sniffer = csv.Sniffer()
        dialect = csv.excel
        with contextlib.suppress(csv.Error):
            dialect = sniffer.sniff(
                "\n".join(decoded_file[: self.SNIFFING_CHARS_COUNT])
            )

        # Read CSV data using detected dialect
        reader = csv.DictReader(decoded_file, dialect=dialect)

        data = list(reader)
        return data
