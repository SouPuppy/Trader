#
# agent_prime:
# 
# Model -> [Action] -> RAG rerank -> [Action] -> RiskControl(Leveraging)
#   ^                                                |
#   |                                           Trade History
#   |                                                |
#   +------------------- Reflection -----------------+
#

Action = {
    Symbol
    Weight: [-1, 1],
}
