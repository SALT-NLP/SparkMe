from typing import Type, Dict, Any, List
from langchain_core.tools import BaseTool
from pydantic import BaseModel
import xml.etree.ElementTree as ET
import json
import re

from src.utils.constants.colors import ORANGE, RESET

def format_tool_as_xml_v2(tool: Type[BaseTool]) -> str:
    """
    Format a tool as XML with a different structure, including newlines for better readability.
    
    :param tool: A class that inherits from BaseTool
    :return: XML-formatted string describing the tool
    """
    lines = []
    lines.append(f"<{tool.name}>")
    lines.append(f"  <description>")
    lines.append(f"    {tool.description}")
    lines.append(f"  </description>")
    
    # Add arguments if present
    if tool.args_schema and issubclass(tool.args_schema, BaseModel):
        lines.append(f"  <arguments>")
        for field_name, field in tool.args_schema.model_fields.items():
            lines.append(f"    <{field_name}>")
            lines.append(f"      <type>{field.annotation.__name__}</type>")
            if field.description:
                lines.append(f"      <description>")
                lines.append(f"        {field.description}")
                lines.append(f"      </description>")
            lines.append(f"    </{field_name}>")
        lines.append(f"  </arguments>")
    
    lines.append(f"</{tool.name}>")
    
    return "\n".join(lines)

def parse_tool_calls(xml_string: str) -> Dict[str, Any]:
    """
    Parse XML tool calls with proper XML entity handling
    """
    # First, identify and escape any < or > within response tags
    xml_string = xml_string.replace('&', '&amp;')
    # xml_string = xml_string.replace('<', '&lt;')
    # xml_string = xml_string.replace('>', '&gt;')
    xml_string = xml_string.replace('"', '&quot;')
    xml_string = xml_string.replace("'", '&apos;')
    
    # Find content between <response> tags and escape < and > within it
    def escape_response_content(match):
        content = match.group(1)
        escaped_content = content.replace('<', '&lt;').replace('>', '&gt;')
        return f"<response>{escaped_content}</response>"
    
    xml_string = re.sub(r'<response>(.*?)</response>', 
                        escape_response_content, 
                        xml_string, 
                        flags=re.DOTALL)
    
    root = ET.fromstring(xml_string)
    result = []
    
    def parse_value(text: str) -> Any:
        """Parse a value that might be a list or other data type."""
        if not text:
            return ""
        text = text.strip()
        
        # Try to parse as a list if it looks like one
        if text.startswith('[') and text.endswith(']'):
            try:
                import ast
                return ast.literal_eval(text)
            except:
                pass
                
        # Try to parse as JSON
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            # If not valid JSON, return as string
            return text
    
    # Iterate through each direct child of tool_calls (each is a tool name)
    for tool_element in root:
        tool_name = tool_element.tag
        arguments = {}
        
        # Each child of the tool element is an argument
        for arg in tool_element:
            arguments[arg.tag] = parse_value(arg.text)
                
        result.append({
            'tool_name': tool_name,
            'arguments': arguments
        })
    
    return result


def call_tool_from_xml(tool_calls_xml_string: str, available_tools: Dict[str, BaseTool]) -> str:
    parsed_calls = parse_tool_calls(tool_calls_xml_string)
    print(f"{ORANGE}Parsed calls:\n{parsed_calls}{RESET}")
    results = []
    
    for call in parsed_calls:
        tool_name = call['tool_name']
        arguments = call['arguments']
        
        if tool_name not in available_tools:
            results.append(f"Error: Tool '{tool_name}' not found.")
            continue
        
        tool = available_tools[tool_name]
        try:
            result = tool._run(**arguments)
            results.append(f"Tool '{tool_name}' executed successfully."
                           f" Result: {result}")
        except Exception as e:
            print(f"Error calling tool '{tool_name}': {str(e)}")
            results.append(f"Error calling tool '{tool_name}': {str(e)}")
    
    return "\n".join(results)

def extract_tool_calls_xml(response: str) -> str:
    """Extract the part of the response containing tool calls."""
    tool_calls_start = response.find("<tool_calls>")
    tool_calls_end = response.find("</tool_calls>")
    if tool_calls_start == -1 or tool_calls_end == -1:
        return ""
    return response[tool_calls_start:tool_calls_end + len("</tool_calls>")]

def clean_malformed_xml(xml_string: str) -> str:
    """Clean malformed XML by removing unmatched tags.
    
    Args:
        xml_string: Input XML string that might have unmatched tags
        
    Returns:
        Cleaned XML string with unmatched tags removed
        
    Example:
        >>> xml = "<a><b>text</c></b></a>"
        >>> clean_malformed_xml(xml)
        "<a><b>text</b></a>"
    """
    # Split the XML into tokens while preserving whitespace
    tokens = []
    current_token = ""
    in_tag = False
    
    for char in xml_string:
        if char == '<':
            if current_token:
                tokens.append(current_token)
            current_token = '<'
            in_tag = True
        elif char == '>':
            current_token += '>'
            tokens.append(current_token)
            current_token = ""
            in_tag = False
        else:
            current_token += char
            
    if current_token:
        tokens.append(current_token)
    
    # Process tags using a stack
    tag_stack = []
    result_tokens = []
    
    for token in tokens:
        if not token.startswith('<'):
            # Not a tag, just add to result
            result_tokens.append(token)
            continue
            
        if token.startswith('</'):
            # Closing tag
            tag_name = token[2:-1].strip()
            
            # Only keep closing tag if it matches the last opening tag
            if tag_stack and tag_stack[-1] == tag_name:
                tag_stack.pop()
                result_tokens.append(token)
            # else skip this unmatched closing tag
            
        elif token.startswith('<'):
            # Opening tag
            tag_name = token[1:-1].strip()
            if not tag_name.startswith('?') and not tag_name.startswith('!'):
                tag_stack.append(tag_name)
            result_tokens.append(token)
    
    return ''.join(result_tokens)

def extract_tool_arguments(response: str, tool_name: str, arg_name: str) -> List[Any]:
    """Extract specific argument values from tool calls in a response."""
    if "<tool_calls>" not in response:
        return []
        
    tool_calls_start = response.find("<tool_calls>")
    tool_calls_end = response.find("</tool_calls>")
    if tool_calls_start == -1 or tool_calls_end == -1:
        return []
        
    # Extract and clean the tool_calls section
    tool_calls_xml = response[
        tool_calls_start:tool_calls_end + len("</tool_calls>")
    ]
    cleaned_xml = clean_malformed_xml(tool_calls_xml)
    
    values = []
    for call in parse_tool_calls(cleaned_xml):
        if call["tool_name"] == tool_name:
            value = call["arguments"].get(arg_name)
            if value:
                # Handle string representation of lists/dicts
                if isinstance(value, str):
                    # First try to parse as JSON
                    try:
                        import json
                        parsed_value = json.loads(value)
                        values.append(parsed_value)
                        continue
                    except json.JSONDecodeError:
                        pass
                    
                    # If JSON parsing fails, check if it's a bracketed list
                    if value.strip().startswith('[') and value.strip().endswith(']'):
                        # Remove brackets and split by comma
                        items = value[1:-1].split(',')
                        # Strip whitespace from each item
                        parsed_list = [item.strip() for item in items]
                        values.append(parsed_list)
                    else:
                        # For other string values
                        values.append(value)
                else:
                    values.append(value)
    
    return values
