import contextlib

import pyeda.boolalg
import pyeda.boolalg.expr
import pyeda.inter
import z3


class LabelSubspace(object):
    def __init__(self, *children):
        assert all(isinstance(c, LabelSubspace) for c in children)
        self.children = children

    def __eq__(self, other):
        if self.any_factory().eq_symbol_is_equivalence_flag:
            return Equivalent(self, other)
        else:
            return self.__class__ == other.__class__ and self.children == other.children

    def __ne__(self, other):
        if self.any_factory().eq_symbol_is_equivalence_flag:
            return Not(Equivalent(self, other))
        else:
            return not (self.__class__ == other.__class__ and self.children == other.children)

    def eq(self, other):
        return Equivalent(self, other)

    def __and__(self, b):
        items = []
        if isinstance(self, And):
            items.extend(self.children)
        else:
            items.append(self)
        if isinstance(b, And):
            items.extend(b.children)
        else:
            items.append(b)
        return And(*items)

    __rand__ = __and__

    def __or__(self, b):
        items = []
        if isinstance(self, Or):
            items.extend(self.children)
        else:
            items.append(self)
        if isinstance(b, Or):
            items.extend(b.children)
        else:
            items.append(b)
        return Or(*items)

    __ror__ = __or__

    def __lshift__(self, b):
        return Implies(b, self)

    def __rshift__(self, b):
        return Implies(self, b)

    __rrshift__ = __lshift__
    __rlshift__ = __rshift__

    def __invert__(self):
        return Not(self)

    def __hash__(self):
        return hash((self.__class__, tuple(hash(c) for c in self.children)))

    def __repr__(self):
        raise NotImplementedError()

    def __str__(self):
        return repr(self)

    def any_factory(self):
        if hasattr(self, 'factory'):
            return self.factory
        for c in self.children:
            factory = c.any_factory()
            if factory:
                return factory

    def to_pyeda(self):
        return self.to_pyeda_()

    def to_z3(self):
        return self.to_z3_()

    def to_cnf(self, factory=None, simplify=True):
        factory = factory or self.any_factory()
        res = factory.from_pyeda(self.to_pyeda().to_cnf())
        if not isinstance(res, And):
            if not isinstance(res, Or):
                return And(Or(res))
            return And(res)
        return res

    def to_dnf(self, factory=None, simplify=True):
        factory = factory or self.any_factory()
        res = factory.from_pyeda(self.to_pyeda().to_dnf())
        if not isinstance(res, Or):
            if not isinstance(res, And):
                return Or(And(res))
            return Or(res)
        return res

    def satisfy_all(self, factory=None, lib="z3"):
        if lib == "z3":
            solver = z3.Solver()
            solver.add(self.to_z3())
            res = solver.check()
            return res == z3.sat
        elif lib == "pyeda":
            factory = factory or self.any_factory()
            all_res = self.to_pyeda().satisfy_all()
            return ({factory[c.name]: v for c, v in res.items()}
                    for res in all_res)
        else:
            raise Exception()

    def to_python_string(self):
        return self.to_python_()

    def vectorize(self):
        expr = self.vectorize_()
        return eval("lambda c: {}".format(expr))

    def to_python(self, is_list=False):
        expr = self.to_python_(is_list=is_list)
        return eval("lambda c: {}".format(expr))

    def vectorize_(self):
        raise NotImplementedError()

    def to_python_(self, **kwargs):
        raise NotImplementedError()

    def to_pyeda_(self):
        raise NotImplementedError()

    def to_z3_(self):
        raise NotImplementedError()

    @property
    def support(self):
        res = set()
        for c in self.children:
            res |= set(c.support)
        return sorted(res)

    @classmethod
    def from_pyeda(cls, expr, factory):
        if isinstance(expr, pyeda.boolalg.expr.AndOp):
            return And(*(cls.from_pyeda(c, factory) for c in expr.xs))
        elif isinstance(expr, pyeda.boolalg.expr.OrOp):
            return Or(*(cls.from_pyeda(c, factory) for c in expr.xs))
        elif isinstance(expr, pyeda.boolalg.expr.NotOp):
            return Not(*(cls.from_pyeda(c, factory) for c in expr.xs))
        elif isinstance(expr, pyeda.boolalg.expr.Complement):
            return Not(cls.from_pyeda(expr.__invert__(), factory))
        elif isinstance(expr, pyeda.boolalg.expr.Variable):
            return factory[expr.name]
        elif isinstance(expr, pyeda.boolalg.expr.Zero):
            return empty
        elif isinstance(expr, pyeda.boolalg.expr.One):
            return full_space
        else:
            raise Exception(f"Unrecognized pyeda object {type(expr)}")

    def __lt__(self, other):
        return repr(self) < repr(other)

class And(LabelSubspace):

    def to_pyeda_(self):
        if len(self.children) > 1:
            return pyeda.inter.And(*(c.to_pyeda_() for c in self.children))
        else:
            return self.children[0].to_pyeda_()

    def to_z3_(self):
        if len(self.children) > 1:
            return z3.And(*(c.to_z3_() for c in self.children))
        else:
            return self.children[0].to_z3_()

    def to_python_(self, **kwargs):
        if len(self.children) > 2:
            return "(all(({})))".format(", ".join(c.to_python_(**kwargs) for c in self.children))
        elif len(self.children) == 2:
            return "({} & {})".format(self.children[0].to_python_(**kwargs), self.children[1].to_python_(**kwargs))
        else:
            return self.children[0].to_python_(**kwargs)

    def vectorize_(self):
        if len(self.children) > 2:
            return "(np.all(({}), axis=0))".format(", ".join(c.vectorize_() for c in self.children))
        elif len(self.children) == 2:
            return "({} & {})".format(self.children[0].vectorize_(), self.children[1].vectorize_())
        else:
            return self.children[0].vectorize_()

    def __repr__(self):
        if len(self.children) > 1:
            return "({})".format(" & ".join(repr(c) for c in self.children))
        return repr(self.children[0])


class Or(LabelSubspace):

    def to_pyeda_(self):
        if len(self.children) > 1:
            return pyeda.inter.Or(*(c.to_pyeda_() for c in self.children))
        else:
            return self.children[0].to_pyeda_()

    def to_z3_(self):
        if len(self.children) > 1:
            return z3.Or(*(c.to_z3_() for c in self.children))
        else:
            return self.children[0].to_z3_()

    def to_python_(self, **kwargs):
        if len(self.children) > 2:
            return "(any(({})))".format(", ".join(c.to_python_(**kwargs) for c in self.children))
        elif len(self.children) == 2:
            return "({} | {})".format(self.children[0].to_python_(**kwargs), self.children[1].to_python_(**kwargs))
        else:
            return self.children[0].to_python_(**kwargs)

    def vectorize_(self):
        if len(self.children) > 2:
            return "(np.any(({}), axis=0))".format(", ".join(c.vectorize_() for c in self.children))
        elif len(self.children) == 2:
            return "({} | {})".format(self.children[0].vectorize_(), self.children[1].vectorize_())
        else:
            return self.children[0].vectorize_()

    def __repr__(self):
        if len(self.children) > 1:
            return "({})".format(" | ".join(repr(c) for c in self.children))
        return repr(self.children[0])


class Not(LabelSubspace):

    def to_pyeda_(self):
        return pyeda.inter.Not(*(c.to_pyeda_() for c in self.children))

    def to_z3_(self):
        return z3.Not(*(c.to_z3_() for c in self.children))

    def to_python_(self, **kwargs):
        return "(~{})".format(self.children[0].to_python_(**kwargs))

    def vectorize_(self):
        return "(~{})".format(self.children[0].vectorize_())

    def __repr__(self):
        return "~{}".format(repr(self.children[0]))


class Implies(LabelSubspace):

    def to_pyeda_(self):
        return pyeda.inter.Implies(*(c.to_pyeda_() for c in self.children))

    def to_z3_(self):
        return z3.Implies(*(c.to_z3_() for c in self.children))

    def to_python_(self, **kwargs):
        return "(~{} | {})".format(self.children[0].to_python_(**kwargs), self.children[1].to_python_(**kwargs))

    def vectorize_(self):
        return "(~{} | {})".format(self.children[0].vectorize_(), self.children[1].vectorize_())

    def __repr__(self):
        return "({} >> {})".format(repr(self.children[0]), repr(self.children[1]))


class Equivalent(LabelSubspace):

    def to_pyeda_(self):
        return pyeda.inter.Equal(*(c.to_pyeda_() for c in self.children))

    def to_z3_(self):
        return self.children[0].to_z3_() == self.children[1].to_z3_()

    def to_python_(self, **kwargs):
        return "({} == {})".format(self.children[0].to_python_(**kwargs), self.children[1].to_python_(**kwargs))

    def vectorize_(self):
        return "({} == {})".format(self.children[0].vectorize_(), self.children[1].vectorize_())

    def __repr__(self):
        return "({} == {})".format(repr(self.children[0]), repr(self.children[1]))


class ITE(LabelSubspace):
    def to_pyeda_(self):
        return pyeda.inter.ITE(*(c.to_pyeda_() for c in self.children))

    def to_z3_(self):
        return z3.If(self.children[0].to_z3_(), self.children[1].to_z3_(), self.children[2].to_z3_())

    def to_python_(self, **kwargs):
        return "(({} & {}) | (~{} & {}))".format(self.children[0].to_python_(**kwargs),
                                                 self.children[1].to_python_(**kwargs),
                                                 self.children[0].to_python_(**kwargs),
                                                 self.children[2].to_python_(**kwargs))

    def vectorize_(self):
        return "(({} & {}) | (~{} & {}))".format(self.children[0].vectorize_(),
                                                 self.children[1].vectorize_(),
                                                 self.children[0].vectorize_(),
                                                 self.children[2].vectorize_())
    def __repr__(self):
        return "({} ? {} : {})".format(repr(self.children[0]), repr(self.children[1]), repr(self.children[2]))


class AtomLabelSubspace(LabelSubspace):
    def __init__(self, uid, index, factory):
        self.index = index
        self.factory = factory
        self.uid = str(uid)
        super().__init__()

    def __hash__(self):
        return hash(self.uid)

    @property
    def absolute_idx(self):
        return self.index

    @property
    def support(self):
        return {self}

    def __eq__(self, other):
        if self.any_factory().eq_symbol_is_equivalence_flag:
            return super().__eq__(other)
        else:
            return other.__class__ == self.__class__ and self.name == other.name

    @property
    def name(self):
        return self.uid

    def to_pyeda_(self):
        return pyeda.inter.exprvar(self.uid)

    def to_python_(self, is_list=False, **kwargs):
        return f"c.get('{self.uid}', False)" if not is_list else f"('{self.uid}' in c)"

    def to_z3_(self):
        return z3.Bool(self.uid)

    def vectorize_(self):
        return f"c[:, {self.index}]"

    def __repr__(self):
        return self.uid


class ExactlyOne(LabelSubspace):
    def to_pyeda_(self):
        return pyeda.inter.OneHot(*[a.to_pyeda_() for a in self.children])

    def to_python_(self, **kwargs):
        return "(sum(({})) == 1)".format(", ".join(c.to_python_(**kwargs) for c in self.children))

    def to_z3_(self):
        return z3.PbEq(*[a.to_z3_() for a in self.children], 1)

    def vectorize_(self):
        return "(np.sum(({}), axis=0) == 1)".format(", ".join(c.vectorize_() for c in self.children))

    def __repr__(self):
        return "(sum({}) == 1)".format(", ".join(repr(c) for c in self.children))


class AtMostOne(LabelSubspace):
    def to_pyeda_(self):
        return pyeda.inter.OneHot0(*[a.to_pyeda_() for a in self.children])

    def to_python_(self, **kwargs):
        return "(sum(({})) <= 1)".format(", ".join(c.to_python_(**kwargs) for c in self.children))

    def to_z3_(self):
        return z3.AtMost(*[a.to_z3_() for a in self.children], 1)

    def vectorize_(self):
        return "(np.sum(({}), axis=0) <= 1)".format(", ".join(c.vectorize_() for c in self.children))

    def __repr__(self):
        return "(sum({}) <= 1)".format(", ".join(repr(c) for c in self.children))


class FullSpace(LabelSubspace):
    def vectorize_(self):
        return "True"

    def to_pyeda_(self):
        return pyeda.inter.expr(True)

    def to_z3_(self):
        return True

    def to_python_(self, **kwargs):
        return "True"

    def __repr__(self):
        return "True"


class Empty(LabelSubspace):
    def vectorize_(self):
        return "False"

    def to_pyeda_(self):
        return pyeda.inter.expr(False)

    def to_z3_(self):
        return False

    def to_python_(self, **kwargs):
        return "False"

    def __repr__(self):
        return "False"


class ClassifierAtom(AtomLabelSubspace):
    @property
    def classifier_idx(self):
        return int(self.uid[1:].split('_')[0])

    @property
    def relative_idx(self):
        return int(self.uid[1:].split('_')[1])


class BratLabelAtom(AtomLabelSubspace):
    @property
    def annotation_type(self):
        if '__' in self.uid:
            return "attribute"
        else:
            return "mention"

    @property
    def annotation_name(self):
        return self.uid.split('__')[0]

    @property
    def value(self):
        if '__' in self.uid:
            return self.uid.split('__')[1]
        return None


class LabelFactory(object):
    def __init__(self):
        self.atoms = {}
        self.last_indexes = [0]
        self.eq_symbol_is_equivalence_flag = False

    @contextlib.contextmanager
    def eq_is_equivalence(self):
        self.eq_symbol_is_equivalence_flag = True
        yield
        self.eq_symbol_is_equivalence_flag = False

    def get_atom(self, name):
        if name not in self.atoms:
            if name.startswith("_"):
                atom = ClassifierAtom(name, self.last_indexes[-1], self)
            else:
                atom = BratLabelAtom(name, self.last_indexes[-1], self)
            self.atoms[name] = atom, self.last_indexes[-1]
            self.last_indexes[-1] += 1
            return atom
        else:
            return self.atoms[name][0]

    def from_pyeda(self, expr):
        return LabelSubspace.from_pyeda(expr, self)

    def __getitem__(self, name):
        return self.get_atom(name)

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError()
        return self.get_atom(name)

    @contextlib.contextmanager
    def new_atoms_set(self):
        self.last_indexes.append(0)
        yield self
        self.last_indexes.pop(-1)


full_space = FullSpace()
empty = Empty()

AtLeastOne = Or
