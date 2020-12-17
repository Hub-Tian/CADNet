from .rpn_head import RPNV2
from .rpn_head import RPNV2_Expand


bbox_head_modules = {
    'RPNV2': RPNV2,
    'RPNV2_Expand': RPNV2_Expand
}