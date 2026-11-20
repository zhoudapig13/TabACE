"""Expose feature coverage in a copy of the recovered historical operator.

Only its top-1/top-2/top-3 feature limits are multiplied. The original recovered
source is left unchanged. Multiplier 1 is checked against the original output.
"""
import ast
import inspect
import recovered_public_shift as original


def _build():
    tree=ast.parse(inspect.getsource(original.apply_public_shift_to_split))
    fn=tree.body[0];fn.name='apply_feature_scaled_shift'
    fn.args.kwonlyargs.append(ast.arg(arg='feature_multiplier',annotation=ast.Name(id='int',ctx=ast.Load())))
    fn.args.kw_defaults.append(ast.Constant(value=1))
    class Limits(ast.NodeTransformer):
        changed=0
        def visit_Call(self,node):
            self.generic_visit(node)
            if (isinstance(node.func,ast.Name) and node.func.id=='min' and len(node.args)==2
                and isinstance(node.args[0],ast.Constant) and node.args[0].value in [1,2,3]
                and isinstance(node.args[1],ast.Name) and node.args[1].id=='n_features'):
                node.args[0]=ast.BinOp(left=node.args[0],op=ast.Mult(),right=ast.Name(id='feature_multiplier',ctx=ast.Load()));self.changed+=1
            return node
    transform=Limits();tree=transform.visit(tree);assert transform.changed==3
    namespace=dict(vars(original));exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),namespace)
    return namespace['apply_feature_scaled_shift']


apply_feature_scaled_shift=_build()
