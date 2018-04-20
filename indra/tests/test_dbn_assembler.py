from indra.assemblers.dbn_assembler import *
from indra.tests.test_cag_assembler import statements

def test_make_model():
    da = DBNAssembler(statements)
    da.make_model()
    assert(da.s_labels == ['inorganic fertilizer', '∂(inorganic fertilizer)/∂t',
        'farm sizes', '∂(farm sizes)/∂t'])
