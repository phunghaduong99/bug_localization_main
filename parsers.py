import csv

import glob
import os.path
from collections import OrderedDict
from datetime import datetime

import javalang
import pygments
import xmltodict
from pygments.lexers import JavaLexer
from pygments.token import Token
from extract_methods import *

class BugReport:
    """Class representing each bug report"""

    __slots__ = ['summary', 'description', 'fixed_files',
                 'pos_tagged_summary', 'pos_tagged_description', 'stack_traces', 'fixdate']

    def __init__(self, summary, description, fixed_files, fixdate):
        self.summary = summary
        self.description = description
        self.fixed_files = fixed_files
        self.pos_tagged_summary = None
        self.pos_tagged_description = None
        self.stack_traces = None
        self.fixdate = fixdate


class SourceFile:
    """Class representing each source file"""

    __slots__ = ['all_content', 'comments', 'class_names', 'attributes',
                 'method_names', 'variables', 'file_name', 'pos_tagged_comments',
                 'exact_file_name', 'package_name', 'methods', 'methods_api']

    def __init__(self, all_content, comments, class_names, attributes,
                 method_names, variables, file_name, package_name, methods, methods_api):
        self.all_content = all_content
        self.comments = comments
        self.class_names = class_names
        self.attributes = attributes
        self.method_names = method_names
        self.variables = variables
        self.file_name = file_name
        self.exact_file_name = file_name[0]
        self.package_name = package_name
        self.pos_tagged_comments = None
        self.methods = methods
        self.methods_api = methods_api


class Parser:
    """Class containing different parsers"""

    __slots__ = ['name', 'src', 'bug_repo']

    def __init__(self, project):
        self.name = project.name
        self.src = project.src
        self.bug_repo = project.bug_repo

    def report_parser(self):
        """Parse XML format bug reports"""

        # Convert XML bug repository to a dictionary
        with open(self.bug_repo) as xml_file:
            xml_dict = xmltodict.parse(
                xml_file.read(), force_list={'file': True})

        # Iterate through bug reports and build their objects
        bug_reports = OrderedDict()

        for bug_report in xml_dict['bugrepository']['bug']:
            bug_reports[bug_report['@id']] = BugReport(
                bug_report['buginformation']['summary'],
                bug_report['buginformation']['description']
                if bug_report['buginformation']['description'] else '',
                [os.path.normpath(path)
                 for path in bug_report['fixedFiles']['file']],
                bug_report['@opendate']
            )

        return bug_reports

    def report_parser_txt(self):
        """Parse XML format bug reports"""

        reader = csv.DictReader(open(self.bug_repo, "r"), delimiter="\t")
        bug_reports = OrderedDict()

        # raw_texts = []
        # fixed_files = []

        for line in reader:
            line["report_time"] = datetime.strptime(line["report_time"], "%Y-%m-%d %H:%M:%S")
            temp = line["files"].strip().split(".java")
            length = len(temp)
            x = []

            for i, f in enumerate(temp):
                f = f.strip()
                if i == (length - 1):
                    continue
                x.append(os.path.normpath(f + ".java"))

            line["files"] = x
            bug_reports[line["bug_id"]] = BugReport(line["summary"], line["description"], line["files"]
                                                    , line["report_time"])

        bug_reports = OrderedDict(sorted(bug_reports.items(), key=lambda item: item[1].fixdate, reverse=True))
        return bug_reports

    def src_parser(self):
        """Parse source code directory of a program and collect
        its java files.
        """

        # Getting the list of source files recursively from the source directory
        src_addresses = glob.glob(str(self.src) + '/**/*.java', recursive=True)

        # Creating a java lexer instance for pygments.lex() method
        java_lexer = JavaLexer()

        src_files = OrderedDict()

        # Looping to parse each source file
        for src_file in src_addresses:
            with open(src_file,  encoding='latin-1') as file:
                src = file.read()

            # Placeholder for different parts of a source file
            comments = ''
            class_names = []
            attributes = []
            method_names = []
            variables = []
            methods = []
            methods_api = []

            # Source parsing
            parse_tree = None
            try:
                parse_tree = javalang.parse.parse(src)
                for path, node in parse_tree.filter(javalang.tree.VariableDeclarator):
                    if isinstance(path[-2], javalang.tree.FieldDeclaration):
                        attributes.append(node.name)
                    elif isinstance(path[-2], javalang.tree.VariableDeclaration):
                        variables.append(node.name)
            except:
                pass

            # Extract methods
            if parse_tree:
                for _, node_to_find in parse_tree.filter(javalang.tree.MethodDeclaration):
                    start, end = get_start_end_for_node(node_to_find, parse_tree)
                    method = removeComments(get_string(start, end, src))
                    methods.append(method)
                    methods_api.append(node_to_find.documentation)

            # Trimming the source file
            ind = False
            if parse_tree:
                if parse_tree.imports:
                    last_imp_path = parse_tree.imports[-1].path
                    src = src[src.index(last_imp_path) +
                              len(last_imp_path) + 1:]
                elif parse_tree.package:
                    package_name = parse_tree.package.name
                    src = src[src.index(package_name) + len(package_name) + 1:]
                else:  # There is no import and no package declaration
                    ind = True
            # javalang can't parse the source file
            else:
                ind = True

            # Lexically tokenize the source file
            lexed_src = pygments.lex(src, java_lexer)

            for i, token in enumerate(lexed_src):
                if token[0] in Token.Comment:
                    if ind and i == 0 and token[0] is Token.Comment.Multiline:
                        src = src[src.index(token[1]) + len(token[1]):]
                        continue
                    comments += token[1]
                elif token[0] is Token.Name.Class:
                    class_names.append(token[1])
                elif token[0] is Token.Name.Function:
                    method_names.append(token[1])

            # Get the package declaration if exists
            if parse_tree and parse_tree.package:
                package_name = parse_tree.package.name
            else:
                package_name = None

            if self.name == 'aspectj' or 'tomcat' or 'eclipse' or 'swt' or 'birt':
                src_files[os.path.relpath(src_file, start=self.src)] = SourceFile(
                    src, comments, class_names, attributes,
                    method_names, variables,
                    [os.path.basename(src_file).split('.')[0]],
                    package_name, methods, methods_api
                )
            else:
                # If source file has package declaration
                if package_name:
                    src_id = (package_name + '.' +
                              os.path.basename(src_file))
                else:
                    src_id = os.path.basename(src_file)

                src_files[src_id] = SourceFile(
                    src, comments, class_names, attributes,
                    method_names, variables,
                    [os.path.basename(src_file).split('.')[0]],
                    package_name, methods, methods_api
                )

        return src_files


def test():
    import datasets

    parser = Parser(datasets.zxing)
    x = parser.report_parser()
    d = parser.src_parser()

    src_id, src = list(d.items())[10]
    print(src_id, src.exact_file_name, src.package_name)


if __name__ == '__main__':
    test()
