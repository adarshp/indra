from __future__ import absolute_import, print_function, unicode_literals
from indra.util import flatten, flatMap, compose, ltake
from pandas import Series, DataFrame, read_csv
from itertools import permutations
import datetime
from scipy.stats import gaussian_kde
import json
import numpy as np
from networkx import DiGraph
from future.utils import lmap, lfilter
from functools import partial
import logging
import os

# Python 2
try:
    basestring
# Python 3
except:
    basestring = str

logger = logging.getLogger('dbn_assembler')

get_respdevs = lambda gb: gb['respdev']
exists = lambda x: True if x is not None else False
deltas = lambda s: (s.subj_delta, s.obj_delta)
get_stdev = lambda x: 1
get_units = lambda x: 'units'

# Location of the CLULab gradable adjectives data.
adjectiveData = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'adjectiveData.tsv')

def _get_adjective(gb, delta):
    """ Get the first adjective from subj_delta or obj_delta """

    if isinstance(delta['adjectives'], list):
        if delta['adjectives']:
            adj = delta['adjectives'][0]
        else:
            adj = None
    else:
        adj = delta['adjectives']

    return adj if adj in gb.groups.keys() else None

def _constructConditionalPDF(simulableStatements, gb, rs):
    # Collect adjectives
    get_adjective = partial(_get_adjective, gb)
    get_adjectives = lambda s: lmap(get_adjective, deltas(s))
    adjs = set(filter(exists, flatMap(get_adjectives, simulableStatements)))

    # Make a adjective-response dict.
    adjectiveResponses = {a: get_respdevs(gb.get_group(a)) for a in adjs}


    responses = lambda a: adjectiveResponses[a] if exists(a) else rs
    delta_responses = lambda d: d['polarity'] * np.array(responses(get_adjective(d)))
    response_tuples = lmap(lambda s: map(delta_responses, deltas(s)), simulableStatements)

    rs_subj, rs_obj = list(*zip(response_tuples))[0]
    xs1, ys1 = np.meshgrid(rs_subj, rs_obj, indexing='xy')

    hasSamePolarity = \
        lambda s: s.subj_delta['polarity'] == s.obj_delta['polarity']

    if len(lmap(hasSamePolarity, simulableStatements)) == 1:
        xs2, ys2 = -xs1, -ys1
        thetas = np.append(
            np.arctan2(ys1.flatten(), xs1.flatten()),
            np.arctan2(ys2.flatten(), xs2.flatten())
        )
    else:
        thetas = np.arctan2(ys1.flatten(), xs1.flatten())

    return gaussian_kde(thetas)

def _attachConditionalProbability(e, gb, rs):
    isSimulable = lambda s: all(map(exists, map(lambda x: x['polarity'],
                                    deltas(s))))
    simulableStatements = lfilter(isSimulable, e[2]['InfluenceStatements'])

    if simulableStatements:
        e[2]['ConditionalProbability'] = \
            _constructConditionalPDF(simulableStatements, gb, rs)
    else:
        e[2]['ConditionalProbability'] = None

def add_conditional_probabilities(CAG):
    # Create a pandas GroupBy object
    gb = read_csv(adjectiveData, delim_whitespace=True).groupby('adjective')

    responses_kde = compose(gaussian_kde, get_respdevs)
    rs = flatMap(lambda g: responses_kde(g[1]).resample(50)[0].tolist(), gb)

    for e in CAG.edges(data=True):
        _attachConditionalProbability(e, gb, rs)

    return CAG



def construct_CAG_skeleton(sts):
    concepts = lambda s: (s.subj.name, s.obj.name)
    makeEdge = lambda p: \
        (p[0], p[1],
                {'InfluenceStatements': lfilter(lambda s: (p[0], p[1]) == concepts(s), sts)})

    edges = map(makeEdge, permutations(set(flatMap(concepts, sts)), 2))
    validEdges = lfilter(lambda t: len(t[2]['InfluenceStatements']) != 0, edges)

    return DiGraph(validEdges)

class DBNAssembler(object):
    """Assembles a dynamic Bayes network from INDRA Statements.

    Parameters
    ----------
    stmts : Optional[list[indra.statement.Statements]]
        A list of INDRA Statements to be assembled. Currently supports
        Influence Statements.

    Attributes
    ----------
    statements : list[indra.statements.Statement]
        A list of INDRA Statements to be assembled.
    dt: float
        The time step
    CAG :
        A networkx DiGraph with conditional probabilities attached

    """
    def __init__(self, stmts=None, dt = None):
        if not stmts:
            self.statements = []
        else:
            self.statements = stmts
        self.dt = dt


    def make_model(self, stdevs = None):
        self.CAG =\
        add_conditional_probabilities(construct_CAG_skeleton(self.statements))

        # Labels of latent state components
        self.s_labels = flatMap(lambda n: (n, '∂({})/∂t'.format(n)),
                                self.CAG.nodes())
        if stdevs is not None:
            self.stdevs = stdevs
        else:
            self.stdevs={x:get_stdev(x) for x in self.s_labels[::2]}


    def sample_transition_function(self):
        A = DataFrame(np.identity(len(self.s_labels)), self.s_labels,
                                  self.s_labels)

        # Initialize certain off-diagonal elements to represent discretized PDE
        # system update.

        for c in self.s_labels[::2]:
            A['∂({})/∂t'.format(c)][c] = 1

        # Sample coefficients from conditional probability data in CAGs
        for e in self.CAG.edges(data=True):
            if exists(e[2]['ConditionalProbability']):
                beta_tilde = np.tan(e[2]['ConditionalProbability'].resample(50)[0][0])
                beta =  (self.stdevs[e[0]]/self.stdevs[e[1]]) * beta_tilde
                pd1, pd2 = ['∂({})/∂t'.format(x) for x in [e[0], e[1]]]
                A[pd1][pd2] = beta * self.dt

        return lambda s: Series(A.as_matrix() @ s.values, s.index)


    def sample_sequence(self, s0, n_steps):
        tf = self.sample_transition_function()
        return ltake(n_steps, iterate(tf, s0))


    def execute_model(self, s0, n_steps, n_samples):
        return [self.sample_sequence(s0, n_steps) for x in trange(n_samples)]


    def export_model(self):
        for n in self.CAG.nodes(data = True):
            n[1]['arguments'] = {"ref": p for p in list(self.CAG.predecessors(n[0]))}
            n[1]['dtype'] = 'real'
            n[1]['units'] = get_units(n[1])

        model = {
            'name' : 'Dynamic Bayes Net Model',
            'createdAt' : str(datetime.datetime.now()),
            'createdBy' : 'INDRA DBN Assembler',
            'modelVariables' : list(self.CAG.nodes(data = True)),
            'edges' : lmap(lambda e: lmap(lambda s: s.to_json(),
                e[2]['InfluenceStatements']), self.CAG.edges(data = True))
        }

        with open('dbn_model.json', 'w') as f:
            f.write(json.dumps(model, indent = 2))
