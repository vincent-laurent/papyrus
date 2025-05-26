# Copyright 2025 Mews Labs
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import copy

from papyrus.engine import extractor


class ExtractorFactory:
    @staticmethod
    def get_processor(file, content, extractor):
        if content == "text":
            return file._extract_text(extractor)
        if content == "tables":
            return file._extract_tables(extractor)
        if content == "all":
            return file._extract_all(extractor)


class File:
    def __init__(self, path):
        self.path = path
        self.pages = defaultdict(lambda: {"text": "", "text_md": "", "tables": []})

    @property
    def tables(self):
        tables: list = []
        for page in self.pages.values():
            tables.extend(page.get("tables", []))
        return tables

    def __text_by_format(self, format=""):
        text: str = ""
        for page in self.pages.values():
            text += page.get("text" + format, page.get("text", ""))
        return text

    @property
    def text(self, format=""):
        return self.__text_by_format()

    @property
    def text_markdown(self, format="") -> str:
        return self.__text_by_format(format="_md")

    def _export_text(self, format):
        parts = []
        for p in sorted(self.pages.keys()):
            page = self.pages[p]
            if format == "md" and page.get("text_md"):
                parts.append(page["text_md"].strip())
            else:
                parts.append(page["text"].strip())
        return "\n\n".join(parts)

    def _export_tables(self):
        parts = []
        for p in sorted(self.pages.keys()):
            page = self.pages[p]
            for i, df in enumerate(page["tables"]):
                parts.append(f"<!-- Page {p} - Table {i+1} -->")
                parts.append(df.to_markdown(index=False))
                parts.append("")
        return "\n\n".join(parts)

    def _export_both(self):

        parts = []
        for p in sorted(self.pages.keys()):
            page = self.pages[p]
            text_content = page["text"].strip()
            if text_content:
                parts.append(text_content)
            for i, df in enumerate(page["tables"]):
                parts.append(f"<!-- Page {p} - Table {i+1} -->")
                parts.append(df.to_markdown(index=False))
            parts.append("")
        return "\n\n".join(parts)

    def export(self, format="text", content="text"):

        assert format in {"text", "md"}, "format must be 'text' or 'md'"
        assert content in {
            "text",
            "tables",
            "both",
        }, "content must be 'text', 'tables', or 'both'"

        if content == "text":
            return self._export_text(format)
        elif content == "tables":
            return self._export_tables()
        elif content == "both":
            return self._export_both()

    def extract(self, extractor: "extractor.BaseExtractor", content="text"):
        ExtractorFactory().get_processor(self, content, extractor)

    def _extract_tables(self, extractor: "extractor.BaseExtractor"):
        for page_number in self.pages:
            self.pages[page_number]["tables"] = []
        previous_file = copy.deepcopy(self)
        extractor.run(self)
        for page_number in self.pages:
            if page_number in previous_file.pages:
                self.pages[page_number]["text"] = copy.deepcopy(
                    previous_file.pages[page_number].get("text", "")
                )
                self.pages[page_number]["text_md"] = copy.deepcopy(
                    previous_file.pages[page_number].get("text_md", "")
                )
            else:
                self.pages[page_number]["text"] = ""
                self.pages[page_number]["text_md"] = ""

        return self

    def _extract_text(self, extractor: "extractor.BaseExtractor"):
        for page_number in self.pages:
            self.pages[page_number]["text"] = ""
            self.pages[page_number]["text_md"] = ""
        previous_file = copy.deepcopy(self)
        extractor.run(self)
        for page_number in self.pages:
            if page_number in previous_file.pages:
                self.pages[page_number]["tables"] = copy.deepcopy(
                    previous_file.pages[page_number].get("tables", [])
                )
            else:
                self.pages[page_number]["tables"] = []

        return self

    def _extract_all(self, extractor: "extractor.BaseExtractor"):
        extractor.run(self)
        return self

    def show_capabilities(self, extractor: "extractor.BaseExtractor"):
        caps = getattr(extractor, "capabilities", set())
        print(
            f"{extractor.__class__.__name__} supports: {', '.join(caps) or 'nothing'}"
        )
