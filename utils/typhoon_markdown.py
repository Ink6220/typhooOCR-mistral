"""
Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: MIT
"""

import re
import base64
from typing import List, Dict, Any

def extract_table_from_html(html_string):
    """Extract and clean table tags from HTML string"""
    try:
        table_pattern = re.compile(r'<table.*?>.*?</table>', re.DOTALL)
        tables = table_pattern.findall(html_string)
        cleaned_tables = [re.sub(r'<table[^>]*>', '<table>', table) for table in tables]
        return '\n'.join(cleaned_tables)
    except Exception as e:
        print(f"extract_table_from_html error: {str(e)}")
        return f"<table><tr><td>Error extracting table: {str(e)}</td></tr></table>"


class MarkdownConverter:
    """Convert structured recognition results to Markdown format"""

    def __init__(self):
        self.heading_levels = {'title': '#', 'sec': '##', 'sub_sec': '###'}
        self.special_labels = {'tab', 'fig', 'title', 'sec', 'sub_sec', 'list', 'formula', 'reference', 'alg'}

    def try_remove_newline(self, text: str) -> str:
        try:
            text = text.strip().replace('-\n', '')
            def is_chinese(char): return '\u4e00' <= char <= '\u9fff'
            lines = text.split('\n')
            processed_lines = []
            for i in range(len(lines)-1):
                current_line, next_line = lines[i].strip(), lines[i+1].strip()
                if current_line:
                    if next_line:
                        if is_chinese(current_line[-1]) and is_chinese(next_line[0]):
                            processed_lines.append(current_line)
                        else:
                            processed_lines.append(current_line + ' ')
                    else:
                        processed_lines.append(current_line + '\n')
                else:
                    processed_lines.append('\n')
            if lines and lines[-1].strip():
                processed_lines.append(lines[-1].strip())
            return ''.join(processed_lines)
        except Exception as e:
            print(f"try_remove_newline error: {str(e)}")
            return text

    def _handle_text(self, text: str) -> str:
        try:
            if not text: return ""
            if text.strip().startswith("\\begin{array}") and text.strip().endswith("\\end{array}"):
                text = "$$" + text + "$$"
            elif ("_{" in text or "^{" in text or "\\" in text) and ("$" not in text) and ("\\begin" not in text):
                text = "$" + text + "$"
            text = self._process_formulas_in_text(text)
            text = self.try_remove_newline(text)
            return text
        except Exception as e:
            print(f"_handle_text error: {str(e)}")
            return text

    def _process_formulas_in_text(self, text: str) -> str:
        try:
            delimiters = [('$$', '$$'), ('\\[', '\\]'), ('$', '$'), ('\\(', '\\)')]
            result = text
            for start_delim, end_delim in delimiters:
                current_pos, processed_parts = 0, []
                while current_pos < len(result):
                    start_pos = result.find(start_delim, current_pos)
                    if start_pos == -1:
                        processed_parts.append(result[current_pos:])
                        break
                    processed_parts.append(result[current_pos:start_pos])
                    end_pos = result.find(end_delim, start_pos + len(start_delim))
                    if end_pos == -1:
                        processed_parts.append(result[start_pos:])
                        break
                    formula_content = result[start_pos + len(start_delim):end_pos]
                    processed_formula = formula_content.replace('\n', ' \\\\ ')
                    processed_parts.append(f"{start_delim}{processed_formula}{end_delim}")
                    current_pos = end_pos + len(end_delim)
                result = ''.join(processed_parts)
            return result
        except Exception as e:
            print(f"_process_formulas_in_text error: {str(e)}")
            return text

    def _remove_newline_in_heading(self, text: str) -> str:
        try:
            def is_chinese(char): return '\u4e00' <= char <= '\u9fff'
            return text.replace('\n', '') if any(is_chinese(char) for char in text) else text.replace('\n', ' ')
        except Exception as e:
            print(f"_remove_newline_in_heading error: {str(e)}")
            return text

    def _handle_heading(self, text: str, label: str) -> str:
        try:
            level = self.heading_levels.get(label, '#')
            text = self._remove_newline_in_heading(text.strip())
            text = self._handle_text(text)
            return f"{level} {text}\n\n"
        except Exception as e:
            print(f"_handle_heading error: {str(e)}")
            return f"# Error processing heading: {text}\n\n"

    def _handle_list_item(self, text: str) -> str:
        return f"- {text.strip()}\n"

    def _handle_figure(self, text: str, section_count: int) -> str:
        try:
            if not text.strip(): return f"![Figure {section_count}](data:image/png;base64,)\n\n"
            if text.startswith("data:image/"):
                return f"![Figure {section_count}]({text})\n\n"
            return f"![Figure {section_count}](data:image/png;base64,{text})\n\n"
        except Exception as e:
            print(f"_handle_figure error: {str(e)}")
            return f"*[Error processing figure: {str(e)}]*\n\n"
    
    def _handle_table(self, text: str) -> str:
        """
        [Final Version] แปลงผลลัพธ์ของโมเดลให้เป็นตาราง HTML ที่สะอาด
        - เน้นการค้นหาและสกัดเอาเฉพาะบล็อก <table> ออกมาจากข้อความทั้งหมด
        """
        try:
            table_style = 'style="width: 100%; border-collapse: collapse; border: 1px solid #ddd;"'
            th_style = 'style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2; text-align: left;"'
            td_style = 'style="border: 1px solid #ddd; padding: 8px;"'
            
            text = text.strip()

            # [ใหม่] ลำดับความสำคัญที่ 1: ค้นหาบล็อก <table> ที่สมบูรณ์ภายในข้อความ
            html_match = re.search(r'<table.*?>.*?</table>', text, re.DOTALL | re.IGNORECASE)
            
            if html_match:
                # ถ้าเจอ ให้ดึงมาเฉพาะส่วนที่เป็น HTML table
                html_table_str = html_match.group(0)
                # ทำความสะอาดและใส่ Style
                clean_html = extract_table_from_html(html_table_str)
                clean_html = clean_html.replace('<table>', f'<table {table_style}>')
                clean_html = clean_html.replace('<th>', f'<th {th_style}>')
                clean_html = clean_html.replace('<td>', f'<td {td_style}>')
                return clean_html + "\n\n"

            # ลำดับความสำคัญที่ 2: ถ้าไม่เจอ HTML ให้ลองหา Markdown Pipe Table
            pipe_table_pattern = re.compile(r'^\s*\|.*\|(?:\n|\r\n?)(?:^\s*\|.*\|(?:\n|\r\n?))+', re.MULTILINE)
            pipe_match = pipe_table_pattern.search(text)
            if pipe_match:
                table_text = pipe_match.group(0)
                lines = [line.strip() for line in table_text.strip().split('\n') if line.strip()]
                if not lines: return ""
                
                html = [f"<table {table_style}>"]
                header_cells = [cell.strip() for cell in lines[0].split('|')][1:-1]
                num_columns = len(header_cells)
                html.append("<thead><tr>")
                for cell in header_cells:
                    html.append(f'<th {th_style}>{cell}</th>')
                html.append("</tr></thead>")
                html.append("<tbody>")
                if len(lines) > 2:
                    for line in lines[2:]:
                        body_cells = [cell.strip() for cell in line.split('|')][1:-1]
                        if len(body_cells) == 1 and num_columns > 1:
                            html.append(f'<tr><td {td_style} colspan="{num_columns}">{body_cells[0]}</td></tr>')
                        else:
                            html.append("<tr>")
                            for i, cell in enumerate(body_cells):
                                if i < num_columns:
                                    html.append(f'<td {td_style}>{cell}</td>')
                            for _ in range(num_columns - len(body_cells)):
                                html.append(f'<td {td_style}></td>')
                            html.append("</tr>")
                html.append("</tbody></table>")
                return "\n".join(html) + "\n\n"

            # ลำดับความสำคัญที่ 3: Fallback สำหรับทุกกรณีที่เหลือ
            return f"<p>{text.replace('\n', '<br>')}</p>\n\n"

        except Exception as e:
            print(f"_handle_table error: {str(e)}")
            return f"*[Error processing table: {str(e)}]*\n\n"


        def _handle_formula(self, text: str) -> str:
            try:
                processed_text = self._process_formulas_in_text(text)
                if '$$' not in processed_text and '\\[' not in processed_text:
                    processed_text = f'$${processed_text}$$'
                return f"{processed_text}\n\n"
            except Exception as e:
                print(f"_handle_formula error: {str(e)}")
                return f"*[Error processing formula: {str(e)}]*\n\n"

    def convert(self, recognition_results: List[Dict[str, Any]]) -> str:
        try:
            markdown_content = []
            for section_count, result in enumerate(recognition_results):
                try:
                    label, text = result.get('label', ''), result.get('text', '').strip()
                    if label == 'fig':
                        markdown_content.append(self._handle_figure(text, section_count))
                        continue
                    if not text: continue
                    if label in {'title', 'sec', 'sub_sec'}:
                        markdown_content.append(self._handle_heading(text, label))
                    elif label == 'list': markdown_content.append(self._handle_list_item(text))
                    elif label == 'tab': markdown_content.append(self._handle_table(text))
                    elif label == 'alg': markdown_content.append(self._handle_algorithm(text))
                    elif label == 'formula': markdown_content.append(self._handle_formula(text))
                    elif label not in self.special_labels:
                        processed_text = self._handle_text(text)
                        markdown_content.append(f"{processed_text}\n\n")
                except Exception as e:
                    print(f"Error processing item {section_count}: {str(e)}")
                    markdown_content.append(f"*[Error processing content]*\n\n")
            return self._post_process(''.join(markdown_content))
        except Exception as e:
            print(f"convert error: {str(e)}")
            return f"Error generating markdown content: {str(e)}"

    def _post_process(self, markdown_content: str) -> str:
        try:
            author_pattern = re.compile(r'\\author\{(.*?)\}', re.DOTALL)
            def process_author_match(match): return self._handle_text(match.group(1))
            markdown_content = author_pattern.sub(process_author_match, markdown_content)
            math_author_pattern = re.compile(r'\$(\\author\{.*?\})\$', re.DOTALL)
            match = math_author_pattern.search(markdown_content)
            if match:
                author_cmd = match.group(1)
                author_content_match = re.search(r'\\author\{(.*?)\}', author_cmd, re.DOTALL)
                if author_content_match:
                    processed_content = self._handle_text(author_content_match.group(1))
                    markdown_content = markdown_content.replace(match.group(0), processed_content)
            markdown_content = re.sub(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', r'**Abstract** \1', markdown_content, flags=re.DOTALL)
            markdown_content = re.sub(r'\\begin\{abstract\}', r'**Abstract**', markdown_content)
            markdown_content = re.sub(r'\\eqno\{\((.*?)\)\}', r'\\tag{\1}', markdown_content)
            markdown_content = markdown_content.replace("\[ \\\\", "$$ \\\\").replace("\\\\ \]", "\\\\ $$")
            replacements = [(r'_ {', r'_{'), (r'^ {', r'^{'), (r'\n{3,}', r'\n\n')]
            for old, new in replacements: markdown_content = re.sub(old, new, markdown_content)
            return markdown_content
        except Exception as e:
            print(f"_post_process error: {str(e)}")
            return markdown_content