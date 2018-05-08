REGISTRY = {}

from .basic_agent import BasicAgentController
from .independent_agents import IndependentMultiagentController
from .coma_agents import COMAMultiAgentController, COMAAgentController
from .pomace_agents import poMACEMultiagentController
from .mcce_agents import MCCEMultiagentController

REGISTRY["basic_ac"] = BasicAgentController
REGISTRY["coma_recursive_ac"] = COMAAgentController
REGISTRY["independent_mac"] = IndependentMultiagentController
REGISTRY["coma_mac"] = COMAMultiAgentController
REGISTRY["pomace_mac"] = poMACEMultiagentController
REGISTRY["mcce_mac"] = MCCEMultiagentController

from .vdn_agents import VDNMultiagentController
REGISTRY["vdn_mac"] = VDNMultiagentController

from .qmix_agents import QMIXMultiagentController
REGISTRY["qmix_mac"] = QMIXMultiagentController