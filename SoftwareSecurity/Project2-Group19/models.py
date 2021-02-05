# Type declarations from https://github.com/jquery/esprima/blob/master/src/nodes.ts
import copy
from abc import abstractmethod, ABC
from dataclasses import dataclass
from random import choice
from string import ascii_uppercase
from typing import Union, List, Dict, Set, Tuple, Optional, Iterable

from dataclasses_json import dataclass_json


ArgumentListElement = Union['Expression', 'SpreadElement']
ArrayExpressionElement = Union['Expression', 'SpreadElement', None]
ArrayPatternElement = Union['AssignmentPattern', 'BindingIdentifier', 'BindingPattern', 'RestElement', None]
BindingPattern = Union['ArrayPattern', 'ObjectPattern']
BindingIdentifier = Union['Identifier']
Declaration = Union['AsyncFunctionDeclaration', 'ClassDeclaration', 'ExportDeclaration',
                    'FunctionDeclaration', 'ImportDeclaration', 'VariableDeclaration']
ExportableDefaultDeclaration = Union['BindingIdentifier', 'BindingPattern', 'ClassDeclaration',
                                     'Expression', 'FunctionDeclaration']
ExportableNamedDeclaration = Union['AsyncFunctionDeclaration', 'ClassDeclaration',
                                   'FunctionDeclaration', 'VariableDeclaration']
ExportDeclaration = Union['ExportAllDeclaration', 'ExportDefaultDeclaration', 'ExportNamedDeclaration']
Expression = Union['ArrayExpression', 'ArrowFunctionExpression',
                   'AssignmentExpression', 'AsyncArrowFunctionExpression',
                   'AsyncFunctionExpression', 'AwaitExpression', 'BinaryExpression',
                   'CallExpression', 'ClassExpression', 'ComputedMemberExpression', 'ConditionalExpression',
                   'Identifier', 'FunctionExpression', 'Literal', 'NewExpression', 'ObjectExpression',
                   'RegexLiteral', 'SequenceExpression', 'StaticMemberExpression', 'TaggedTemplateExpression',
                   'ThisExpression', 'UnaryExpression', 'UpdateExpression', 'YieldExpression']
FunctionParameter = Union['AssignmentPattern', 'BindingIdentifier', 'BindingPattern']
ImportDeclarationSpecifier = Union['ImportDefaultSpecifier', 'ImportNamespaceSpecifier', 'ImportSpecifier']
ObjectExpressionProperty = Union['Property', 'SpreadElement']
ObjectPatternProperty = Union['Property', 'RestElement']
Statement = Union['AsyncFunctionDeclaration', 'BreakStatement', 'ContinueStatement',
                  'DebuggerStatement', 'DoWhileStatement', 'EmptyStatement',
                  'ExpressionStatement', 'Directive', 'ForStatement', 'ForInStatement',
                  'ForOfStatement', 'FunctionDeclaration', 'IfStatement', 'ReturnStatement',
                  'SwitchStatement', 'ThrowStatement', 'TryStatement', 'VariableDeclaration',
                  'WhileStatement', 'WithStatement']
PropertyKey = Union['Identifier', 'Literal']
PropertyValue = Union['AssignmentPattern', 'AsyncFunctionExpression', 'BindingIdentifier',
                      'BindingPattern', 'FunctionExpression']
StatementListItem = Union['Declaration', 'Statement']

AssignmentOperators = ['=', '*=', '**=', '/=', '%=', '+=', '-=', '<<=', '>>=', '>>>=', '&=', '^=', '|=']
BinaryExpressionOperators = ['instanceof', 'in', '+', '-', '*', '/', '%', '**', '|', '^', '&', '==',
                             '!=', '===', '!==', '<', '>', '<=', '<<', '>>', '>>>']


@dataclass
class Pattern:
    vulnerability: str      # "DOM XSS"
    sources: Tuple[str]     # ["document.referrer", "document.url", "document.location"]
    sinks: Tuple[str]       # ["eval", "document.write", "document.innerHTML", "setAttribute"]
    sanitizers: Tuple[str]  # ["encodeURI"]


@dataclass(frozen=True)
class Vulnerability:
    vulnerability: str      # "DOM XSS"
    source: Tuple[str]      # ["document.URL"]
    sink: Tuple[str]        # ["document.write"]
    sanitizer: Tuple[str]   # ["encodeURI"]

    def __repr__(self):
        return str(self.__dict__)


class State:
    def __init__(self, patterns: List[Pattern]):
        self.patterns: List[Pattern] = patterns
        # Patterns given as an input

        self.tainted_vars: Dict[Tuple[str, str, Tuple[str, ...]], List[Set[str]]] = {}
        # (variable_name, vulnerability_name, source) -> set of sanitizers that were applied
        # (can be multiple sanitizer sets in case of branching)

        self.vulnerabilities: List[Vulnerability] = []

        self.shadowing_prefixes: List[str] = []
        self.variables: List[str] = []

        # A terrible implementation for implicit flows
        # Taint everything is none if we don't want to taint everything
        #  and is list[(vulnerability, sources, sanitizers)] otherwise
        self.taint_everything: Optional[List[Tuple[str, Tuple[str, ...], List[Set[str]]]]] = None

    def __repr__(self):
        return str(self.__dict__)

    def is_sanitizer_for_vuln(self, vuln: str, sanitizer: str) -> bool:
        for p in self.patterns:
            if p.vulnerability == vuln and sanitizer in p.sanitizers:
                return True
        return False

    def taint_var(self, var: str, taints: List[Tuple[str, Tuple[str, ...], List[Set[str]]]]):
        for vuln, source, sanitizers in taints:
            self.add_sanitizers(var, vuln=vuln, source=source, sanitizers=sanitizers)

    def reset_taintedness(self, var: str):
        tains = self.get_taints(var)
        for vuln, source, _ in tains:
            self.tainted_vars.pop((var, vuln, source))

    def get_taints(self, var: str) -> List[Tuple[str, Tuple[str, ...], List[Set[str]]]]:
        """
        :param var: variable name
        :return: list: [(vulnerability, tuple(sources), list(set(sanitizers))]
        """
        res: List[Tuple[str, Tuple[str, ...], List[Set[str]]]] = []
        # In case `var` is already tainted
        for (var_name, vuln, source), sanitizers in self.tainted_vars.items():
            if var == var_name:
                res.append((vuln, source, sanitizers))

        # In case `var` is a source
        for p in self.patterns:
            if var in p.sources:
                res.append((p.vulnerability, (var,), [set()]))
        return res

    def add_vulnerability_if_sink(self, var: str, tainted_vars: List[Tuple[str, Tuple[str, ...], List[Set[str]]]]):
        # print(f'Checking {var} for being a sink...', self.patterns)
        for p in self.patterns:
            if var not in p.sinks:
                continue
            # print(var, 'IS A SINK!', tainted_vars)
            for vuln, source, sanitizer in tainted_vars:
                for s in sanitizer:
                    vulnerability = Vulnerability(vulnerability=vuln, source=source, sink=(var,), sanitizer=tuple(s))
                    # print('ADDING VULNERABILITY:', vulnerability)
                    self.vulnerabilities.append(vulnerability)

    def add_sanitizers(self, var: str, vuln: str, source: Tuple[str, ...],
                       sanitizers: Iterable[Set[str]], intersection: bool = True):
        """
        Add sanitizer to the self.tainted_vars for (var, vuln, source)
        """
        sanitizers = [set([s for s in st if self.is_sanitizer_for_vuln(vuln, s)])
                      for st in sanitizers]

        key = (var, vuln, source)
        if key not in self.tainted_vars:
            self.tainted_vars[key] = list(copy.deepcopy(sanitizers))
        for target_sanitizer in self.tainted_vars[key]:
            for source_sanitizer in sanitizers:
                if intersection:    target_sanitizer.intersection_update(source_sanitizer)
                else:               target_sanitizer |= source_sanitizer
        uniques = set([tuple(v) for v in self.tainted_vars[key]])
        self.tainted_vars[key] = [set(item) for item in uniques]


class BaseVulnerabilityTracker(ABC):
    @abstractmethod
    def execute(self, state: State) -> str:
        pass


@dataclass_json
@dataclass
class Property:
    type: str
    key: PropertyKey
    computed: bool
    value: Union[PropertyValue, None]
    kind: str
    method: bool
    shorthand: bool


@dataclass_json
@dataclass
class Identifier(BaseVulnerabilityTracker):
    type: str
    name: str

    def execute(self, state: State) -> str:
        for p in state.shadowing_prefixes[::-1]:
            res = f'{p}{self.name}'
            if res in state.variables:
                if state.taint_everything:
                    state.taint_var(res, state.taint_everything)
                #print(f'Identifier: {res}')
                return res

        if state.taint_everything:
            state.taint_var(self.name, state.taint_everything)
        #print(f'Identifier: {self.name}')
        return self.name


@dataclass_json
@dataclass
class Import:
    type: str


@dataclass_json
@dataclass
class Literal(BaseVulnerabilityTracker):
    type: str
    value: Union[bool, int, str, None]
    raw: str

    def execute(self, state: State) -> str:
        if state.taint_everything:
            state.taint_var(self.raw, state.taint_everything)
        #print(f'Literal: {self.raw}')
        return self.raw


@dataclass_json
@dataclass
class BlockStatement(BaseVulnerabilityTracker):
    type: str
    body: List[Statement]

    @staticmethod
    def generate_random_prefix(length: int) -> str:
        return ''.join(choice(ascii_uppercase) for _ in range(length))

    def execute(self, state: State) -> str:
        state.shadowing_prefixes.append(self.generate_random_prefix(length=10))
        res = [s.execute(state) for s in self.body]
        #print('BlockStatement:', res)
        state.shadowing_prefixes.pop()
        return '\n'.join(res)


@dataclass_json
@dataclass
class ArrayExpression:
    type: str
    elements: List[ArrayExpressionElement]


@dataclass_json
@dataclass
class ArrayPattern:
    type: str
    elements: List[ArrayPatternElement]


@dataclass_json
@dataclass
class ArrowFunctionExpression:
    type: str
    id: Union[Identifier, None]
    params: List[FunctionParameter]
    body: Union[BlockStatement, Expression]
    generator: bool
    expression: bool
    is_async: bool


@dataclass_json
@dataclass
class AssignmentExpression(BaseVulnerabilityTracker):
    type: str
    operator: str
    left: Expression
    right: Expression

    def execute(self, state: State) -> str:
        #  (when variable is assigned a sink or when a sink is assigned a safe variable)
        assert self.operator in AssignmentOperators
        l = self.left.execute(state)
        r = self.right.execute(state)

        l_taints = state.get_taints(l)
        r_taints = state.get_taints(r)
        state.add_vulnerability_if_sink(l, tainted_vars=r_taints)

        # make l a safe variable
        for vuln, source, _ in l_taints:
            state.tainted_vars.pop((l, vuln, source), None)

        # Change state in case r is tainted
        for vuln, source, sanitizers in r_taints:
            state.tainted_vars[(l, vuln, source)] = copy.deepcopy(sanitizers)

        #print(f'AssignmentExpression: {l} = {r}')
        #print('TaintedState:', state.tainted_vars)
        #print('State:', state)
        return l


@dataclass_json
@dataclass
class AssignmentPattern:
    type: str
    left: Union[BindingIdentifier, BindingPattern]
    right: Expression


@dataclass_json
@dataclass
class AsyncArrowFunctionExpression:
    type: str
    id: Union[Identifier, None]
    params: List[FunctionParameter]
    body: Union[BlockStatement, Expression]
    generator: bool
    expression: bool
    is_async: bool


@dataclass_json
@dataclass
class AsyncFunctionDeclaration:
    type: str
    id: Union[Identifier, None]
    params: List[FunctionParameter]
    body: BlockStatement
    generator: bool
    expression: bool
    is_async: bool


@dataclass_json
@dataclass
class AsyncFunctionExpression:
    type: str
    id: Union[Identifier, None]
    params: List[FunctionParameter]
    body: BlockStatement
    generator: bool
    expression: bool
    is_async: bool


@dataclass_json
@dataclass
class AwaitExpression:
    type: str
    argument: Expression


@dataclass_json
@dataclass
class BinaryExpression(BaseVulnerabilityTracker):
    type: str
    operator: str
    left: Expression
    right: Expression

    def execute(self, state: State) -> str:
        assert self.operator in BinaryExpressionOperators
        l = self.left.execute(state)
        r = self.right.execute(state)
        l_taints = state.get_taints(l)
        r_taints = state.get_taints(r)
        res = f'{l}{self.operator}{r}'

        # taint of res is the union of l and r
        for vuln, source, sanitizer in l_taints + r_taints:
            key = (res, vuln, source)
            if key not in state.tainted_vars:
                state.tainted_vars[key] = copy.deepcopy(sanitizer)
            for target_sanitizer in state.tainted_vars[key]:
                for source_sanitizer in sanitizer:
                    target_sanitizer.intersection_update(source_sanitizer)
        #print('Binary Expression:', l, self.operator, r)
        #print('TaintedState:', state.tainted_vars)
        return res


@dataclass_json
@dataclass
class BreakStatement(BaseVulnerabilityTracker):
    type: str
    label: Union[Identifier, None]

    def execute(self, state: State) -> str:
        if self.label:
            l = self.label.execute(state)
            res = f'break:{l}'
        else:
            res = 'break'
        return res


@dataclass_json
@dataclass
class CallExpression(BaseVulnerabilityTracker):
    type: str
    callee: Union[Expression, Import]
    arguments: List[ArgumentListElement]

    def execute(self, state: State) -> str:
        c = self.callee.execute(state)
        args = [a.execute(state) for a in self.arguments]
        res = f'{c}({args})'
        state.reset_taintedness(res)
        #print('TaintedState before call:', state.tainted_vars)

        c_taints = state.get_taints(c)
        args_taints = [t for a in args for t in state.get_taints(a)]
        callee_is_source = any(c in p.sources for p in state.patterns)

        #print('C taints:', c_taints)
        #print('Args taints:', args_taints)
        #print('Callee is source:', callee_is_source)

        if len(c_taints) == 0:
            for arg_vuln, arg_source, arg_sanitizer in args_taints:
                state.add_sanitizers(var=res, vuln=arg_vuln, source=arg_source, sanitizers=arg_sanitizer)

        for c_vuln, c_source, c_sanitizer in c_taints:
            state.add_sanitizers(var=res, vuln=c_vuln, source=c_source, sanitizers=c_sanitizer)
            for arg_vuln, arg_source, arg_sanitizer in args_taints:
                # c might have different vuln from args
                if c_vuln != arg_vuln:
                    state.add_sanitizers(var=res, vuln=arg_vuln, source=arg_source, sanitizers=arg_sanitizer)
                    continue

                if callee_is_source and c not in c_source:
                    c_source = (c,) + c_source
                combined_sources = arg_source + c_source
                state.tainted_vars.pop((res, c_vuln, c_source), None)

                state.add_sanitizers(var=res, vuln=arg_vuln, source=combined_sources, sanitizers=arg_sanitizer)
                state.add_sanitizers(var=res, vuln=arg_vuln, source=combined_sources, sanitizers=c_sanitizer)

        # Callee is a sanitizer
        res_taints = state.get_taints(res)
        for p in state.patterns:
            if c not in p.sanitizers:
                continue

            # For all the sources
            for vuln, source, sanitizer in res_taints:
                state.add_sanitizers(var=res, vuln=p.vulnerability, source=source, sanitizers=[{c}], intersection=False)
        #print('TaintedState in the end:', state.tainted_vars)
        #print('CallExpression', res)

        # Callee is a sink
        state.add_vulnerability_if_sink(c, tainted_vars=args_taints)
        return res


@dataclass_json
@dataclass
class CatchClause:
    type: str
    param: Union[BindingIdentifier, BindingPattern]
    body: BlockStatement


@dataclass_json
@dataclass
class ClassBody:
    type: str
    body: List[Property]


@dataclass_json
@dataclass
class ClassDeclaration:
    type: str
    id: Union[Identifier, None]
    superClass: Union[Identifier, None]
    body: ClassBody


@dataclass_json
@dataclass
class ClassExpression:
    type: str
    id: Union[Identifier, None]
    superClass: Union[Identifier, None]
    body: ClassBody


@dataclass_json
@dataclass
class ConditionalExpression:
    type: str
    test: Expression
    consequent: Expression
    alternate: Expression


@dataclass_json
@dataclass
class ContinueStatement:
    type: str
    label: Union[Identifier, None]


@dataclass_json
@dataclass
class DebuggerStatement:
    type: str


@dataclass_json
@dataclass
class Directive:
    type: str
    expression: Expression
    directive: str


@dataclass_json
@dataclass
class DoWhileStatement:
    type: str
    body: Statement
    test: Expression

    def execute(self, state: State) -> str:
        # Run twice to make sure the variables come in sync (Probably not a good way)
        self.body.execute(state)
        unrolled = IfStatement(type='IfStatement', test=self.test, consequent=self.body, alternate=None)
        res = unrolled.execute(state)
        res = unrolled.execute(state)
        return res


@dataclass_json
@dataclass
class EmptyStatement:
    type: str


@dataclass_json
@dataclass
class ExportAllDeclaration:
    type: str
    source: Literal


@dataclass_json
@dataclass
class ExportDefaultDeclaration:
    type: str
    declaration: ExportableDefaultDeclaration


@dataclass_json
@dataclass
class ExportSpecifier:
    type: str
    exported: Identifier
    local: Identifier


@dataclass_json
@dataclass
class ExportNamedDeclaration:
    type: str
    declaration: Union[ExportableNamedDeclaration, None]
    specifiers: List[ExportSpecifier]
    source: Union[Literal, None]


@dataclass_json
@dataclass
class ExpressionStatement(BaseVulnerabilityTracker):
    type: str
    expression: Expression

    def execute(self, state: State) -> str:
        res = self.expression.execute(state)
        #print('ExpressionStatement:', res)
        return res


@dataclass_json
@dataclass
class ForInStatement:
    type: str
    left: Expression
    right: Expression
    body: Statement
    each: bool


@dataclass_json
@dataclass
class ForOfStatement:
    type: str
    is_await: bool
    left: Expression
    right: Expression
    body: Statement


@dataclass_json
@dataclass
class ForStatement:
    type: str
    init: Union[Expression, None]
    test: Union[Expression, None]
    update: Union[Expression, None]
    body: Statement


@dataclass_json
@dataclass
class FunctionDeclaration:
    type: str
    id: Union[Identifier, None]
    params: List[FunctionParameter]
    body: BlockStatement
    generator: bool
    expression: bool
    is_async: bool

    def execute(self, state: State) -> str:
        res = self.body.execute(state)
        return res


@dataclass_json
@dataclass
class FunctionExpression:
    type: str
    id: Union[Identifier, None]
    params: List[FunctionParameter]
    body: BlockStatement
    generator: bool
    expression: bool
    is_async: bool


@dataclass_json
@dataclass
class IfStatement(BaseVulnerabilityTracker):
    type: str
    test: Expression
    consequent: Statement
    # alternate: Union[Statement, None] # changed this to be optional
    alternate: Optional[Union[Statement, None]] = None

    def execute(self, state: State) -> str:
        t = self.test.execute(state)
        consequence_state = copy.deepcopy(state)
        alternate_state = copy.deepcopy(state)

        # Handle implicit flow (23, 24, 49) by checking if the condition is tainted
        #  if(tainted) => all {consequence != alternate} have to be tainted
        test_taints = state.get_taints(t)
        #print('If test taints:', test_taints)
        if len(test_taints) > 0:
            consequence_state.taint_everything = test_taints
            alternate_state.taint_everything = test_taints

        res = f'if({t})'
        if t != '0' and t != 'false':
            c = self.consequent.execute(consequence_state)
            res += f'=>{c}'
        if t != '1' and t != 'true' and self.alternate:
            a = self.alternate.execute(alternate_state)
            res += f' else=>{a}'

        if t == '0' or t == 'false':
            state.tainted_vars = alternate_state.tainted_vars
            return res

        if t == '1' or t == 'true':
            state.tainted_vars = consequence_state.tainted_vars
            return res

        # Unite consequence_state and alternate_states
        state.tainted_vars = consequence_state.tainted_vars
        for k, v in alternate_state.tainted_vars.items():
            if k in state.tainted_vars:
                state.tainted_vars[k] += v
            else:
                state.tainted_vars[k] = v
            uniques = set([tuple(v) for v in state.tainted_vars[k]])
            state.tainted_vars[k] = [set(item) for item in uniques]

        state.vulnerabilities = list(set(consequence_state.vulnerabilities + alternate_state.vulnerabilities))
        #print('IfStatement:', res)
        #print('if taint:', consequence_state.tainted_vars)
        #print('else taint:', alternate_state.tainted_vars)
        #print('TaintedState:', state.tainted_vars)
        return res


@dataclass_json
@dataclass
class ImportDeclaration:
    type: str
    specifiers: List[ImportDeclarationSpecifier]
    source: Literal


@dataclass_json
@dataclass
class ImportDefaultSpecifier:
    type: str
    local: Identifier


@dataclass_json
@dataclass
class ImportNamespaceSpecifier:
    type: str
    local: Identifier


@dataclass_json
@dataclass
class ImportSpecifier:
    type: str
    local: Identifier
    imported: Identifier


@dataclass_json
@dataclass
class LabeledStatement:
    type: str
    label: Identifier
    body: Statement


@dataclass_json
@dataclass
class MetaProperty:
    type: str
    meta: Identifier
    property: Identifier


@dataclass_json
@dataclass
class MethodDefinition:
    type: str
    key: Union[Expression, None]
    computed: bool
    value: Union[AsyncFunctionExpression, FunctionExpression, None]
    kind: str
    static: bool


@dataclass_json
@dataclass
class Module:
    type: str
    body: List[StatementListItem]
    sourceType: str


@dataclass_json
@dataclass
class NewExpression:
    type: str
    callee: Expression
    arguments: List[ArgumentListElement]


@dataclass_json
@dataclass
class ObjectExpression:
    type: str
    properties: List[ObjectExpressionProperty]


@dataclass_json
@dataclass
class ObjectPattern:
    type: str
    properties: List[ObjectPatternProperty]


@dataclass_json
@dataclass
class Regex:
    pattern: str
    flags: str


@dataclass_json
@dataclass
class RegexLiteral:
    type: str
    value: str
    raw: str
    regex: Regex


@dataclass_json
@dataclass
class RestElement:
    type: str
    argument: Union[BindingIdentifier, BindingPattern]


@dataclass_json
@dataclass
class ReturnStatement:
    type: str
    argument: Union[Expression, None]


@dataclass_json
@dataclass
class Script(BaseVulnerabilityTracker):
    type: str
    body: List[StatementListItem]
    sourceType: str
    _comment: Optional[str] = None

    def execute(self, state: State) -> str:
        state.shadowing_prefixes.append(BlockStatement.generate_random_prefix(length=10))
        res = [i.execute(state) for i in self.body]
        #print('Script:', res)
        return '\n'.join(res)


@dataclass_json
@dataclass
class SequenceExpression:
    type: str
    expressions: List[Expression]


@dataclass_json
@dataclass
class SpreadElement:
    type: str
    argument: Expression


@dataclass_json
@dataclass
class StaticMemberExpression(BaseVulnerabilityTracker):
    type: str
    computed: bool
    object: Expression
    property: Expression

    def execute(self, state: State) -> str:
        o = self.object.execute(state)
        p = self.property.execute(state)
        res = f'{o}.{p}'

        # make o.p tainted if o is tainted / is a source
        o_taints = state.get_taints(o)
        for vuln, source, sanitizers in o_taints:
            state.add_sanitizers(var=res, vuln=vuln, source=source, sanitizers=sanitizers)

        #print(f'StaticMemberExpression: {res}', state)
        return res


@dataclass_json
@dataclass
class Super:
    type: str


@dataclass_json
@dataclass
class SwitchCase:
    type: str
    test: Union[Expression, None]
    consequent: List[Statement]


@dataclass_json
@dataclass
class SwitchStatement:
    type: str
    discriminant: Expression
    cases: List[SwitchCase]


@dataclass_json
@dataclass
class TemplateElementValue:
    cooked: str
    raw: str


@dataclass_json
@dataclass
class TemplateElement:
    type: str
    value: TemplateElementValue
    tail: bool


@dataclass_json
@dataclass
class TemplateLiteral:
    type: str
    quasis: List[TemplateElement]
    expressions: List[Expression]


@dataclass_json
@dataclass
class TaggedTemplateExpression:
    type: str
    tag: Expression
    quasi: TemplateLiteral


@dataclass_json
@dataclass
class ThisExpression:
    type: str


@dataclass_json
@dataclass
class ThrowStatement:
    type: str
    argument: Expression


@dataclass_json
@dataclass
class TryStatement:
    type: str
    block: BlockStatement
    handler: Union[CatchClause, None]
    finalizer: Union[BlockStatement, None]


@dataclass_json
@dataclass
class UnaryExpression:
    type: str
    operator: str
    argument: Expression
    prefix: bool


@dataclass_json
@dataclass
class UpdateExpression(BaseVulnerabilityTracker):
    type: str
    operator: str
    argument: Expression
    prefix: bool

    def execute(self, state: State) -> str:
        a = self.argument.execute(state)
        res = f'{self.operator}{a}'
        return res


@dataclass_json
@dataclass
class VariableDeclarator(BaseVulnerabilityTracker):
    type: str
    id: Union[BindingIdentifier, BindingPattern]
    init: Union[Expression, None]
    shadow: bool = False

    def execute(self, state: State) -> str:
        id_res = self.id.execute(state)
        if self.shadow:
            for p in state.shadowing_prefixes[::-1]:
                if p in id_res:
                    id_res = id_res.replace(p, '')
            id_res = state.shadowing_prefixes[-1] + id_res
        res = id_res
        state.variables.append(res)

        if not self.init:
            return res

        res_expression = AssignmentExpression(type='AssignmentExpression', operator='=',
                                              left=Identifier(type='Identifier', name=id_res),
                                              right=self.init)

        res = res_expression.execute(state)
        #print('VariableDeclarator:', res)
        return res


@dataclass_json
@dataclass
class VariableDeclaration(BaseVulnerabilityTracker):
    type: str
    declarations: List[VariableDeclarator]
    kind: str

    def execute(self, state: State) -> str:
        # Handle variable shadowing
        declarations = []
        for dec in self.declarations:
            dec.shadow = True if self.kind in {'let', 'const'} else False
            declarations.append(dec.execute(state))
        res = f'{self.kind} {declarations}'
        #print(f'VariableDeclaration: {res}')
        return res


@dataclass_json
@dataclass
class WhileStatement(BaseVulnerabilityTracker):
    type: str
    test: Expression
    body: Statement

    def execute(self, state: State) -> str:
        # Run twice to make sure the variables come in sync (Probably not a good way)
        unrolled = IfStatement(type='IfStatement', test=self.test, consequent=self.body, alternate=None)
        res = unrolled.execute(state)
        res = unrolled.execute(state)
        return res


@dataclass_json
@dataclass
class WithStatement:
    type: str
    object: Expression
    body: Statement


@dataclass_json
@dataclass
class YieldExpression:
    type: str
    argument: Union[Expression, None]
    delegate: bool


# Aliases
ComputedMemberExpression = StaticMemberExpression
MemberExpression = StaticMemberExpression
Program = Script
