# The script that is needed to get the data is:
"""
pip install xeno-canto
xeno-canto -dl grp:"bats" q:"A"
"""
# grp: bats = stands for the group of animals we are filtering on
# q: A = stands for the quality of the recording. The rank starts from A (clear and loud) to E (unclear)
# Therefore, if we want also other qualities, we just select the same group but a different quality