import copy
from csp import CSP


def create_n_queens_csp(n=8):
    """Create an N-Queen problem on the board of size n * n.

    You should call csp.add_variable() and csp.add_binary_factor().

    Args:
        n: int, number of queens, or the size of one dimension of the board.

    Returns
        csp: A CSP problem with correctly configured factor tables
        such that it can be solved by a weighted CSP solver
    """
    csp = CSP()
    # TODO: Problem b
    # TODO: BEGIN_YOUR_CODE

    # refer the formulation 2
    # there n variables in TOTAL, each one represents the Position of queen in row Qi
    for i in range(n):  # add all the variables to CSP
        csp.add_variable(var=i , domain=list(range(n)))
        # 添加一元约束
        csp.add_unary_factor(var=i,factor_function=lambda x:1)    # factor_function作用在变量var的每个值value上
    # def binaryConstraints(value1,value2):
    #     return abs(value1-value2)

    # 添加二元约束
    for i in range(n):
        for j in range(n):
            if i==j:
                continue
            # factor_function分别作用在变量var1和var2的每个值value1,value2上
            csp.add_binary_factor(var1=i,var2=j,factor_function=lambda x,y:abs(x-y) != abs(i-j)) # 两个皇后不在对角线上
            csp.add_binary_factor(var1=i,var2=j,factor_function=lambda x,y: x != y)              # 两个皇后不在同一列上


    # raise NotImplementedError
    # TODO: END_YOUR_CODE
    return csp


class BacktrackingSearch:
    """A backtracking algorithm that solves CSP.

    Attributes:
        num_assignments: keep track of the number of assignments
            (identical when the CSP is unweighted)
        num_operations: keep track of number of times backtrack() gets called
        first_assignment_num_operations: keep track of number of operations to
            get to the very first successful assignment (maybe not optimal)
        all_assignments: list of all solutions found

        csp: a weighted CSP to be solved
        mcv: bool, if True, use Most Constrained Variable heuristics
        ac3: bool, if True, AC-3 will be used after each variable is made
        domains: dictionary of domains of every variable in the CSP

    Usage:
        search = BacktrackingSearch()
        search.solve(csp)
    """

    def __init__(self):
        self.num_assignments = 0
        self.num_operations = 0
        self.first_assignment_num_operations = 0
        self.all_assignments = []

        self.csp = None
        self.mcv = False
        self.ac3 = False
        self.domains = {}

    def reset_results(self):
        """Resets the statistics of the different aspects of the CSP solver."""
        self.num_assignments = 0
        self.num_operations = 0
        self.first_assignment_num_operations = 0
        self.all_assignments = []

    def check_factors(self, assignment, var, val):
        """Check consistency between current assignment and a new variable.

        Given a CSP, a partial assignment, and a proposed new value for a
        variable, return the change of weights after assigning the variable
        with the proposed value.

        Args:
            assignment: A dictionary of current assignment.
                Unassigned variables do not have entries, while an assigned
                variable has the assigned value as value in dictionary.
                e.g. if the domain of the variable A is [5,6],
                and 6 was assigned to it, then assignment[A] == 6.
            var: name of an unassigned variable.
            val: the proposed value.

        Returns:
            bool
                True if the new variable with value can satisfy constraint,
                otherwise, False
        """
        assert var not in assignment
        if self.csp.unary_factors[var]:
            if self.csp.unary_factors[var][val] == 0:   # check_factors 首先检查一元约束，若对应值为0，则说明该变量不能取值value
                return False
        for var2, factor in self.csp.binary_factors[var].items(): # 检查二元约束 - 找出var的所有邻居，并得到var和邻居var2对应的factor
            if var2 not in assignment:              # 邻居var2还没有赋值，不用考虑var与var2之间的约束关系
                continue
            if factor[val][assignment[var2]] == 0:  # var2已有赋值，则检查它们俩之间对应取值是否矛盾
                return False
        return True

    def solve(self, csp, mcv=False, ac3=False):
        """Solves the given unweighted CSP using heuristics.

        Note that we want this function to find all possible assignments.
        The results are stored in the variables described in
            reset_result().

        Args:
            csp: A unweighted CSP.
            mcv: bool, if True, Most Constrained Variable heuristics is used.
            ac3: bool, if True, AC-3 will be used after each assignment of an
            variable is made.
        """
        self.csp = csp
        self.mcv = mcv
        self.ac3 = ac3
        self.reset_results()
        self.domains = {var: list(self.csp.values[var])
                        for var in self.csp.variables}
        self.backtrack({})

    def backtrack(self, assignment):
        """Back-tracking algorithms to find all possible solutions to the CSP.

        Args:
            assignment: a dictionary of current assignment.
                Unassigned variables do not have entries, while an assigned
                variable has the assigned value as value in dictionary.
                    e.g. if the domain of the variable A is [5, 6],
                    and 6 was assigned to it, then assignment[A] == 6.
        """
        self.num_operations += 1

        num_assigned = len(assignment.keys())
        if num_assigned == self.csp.vars_num:
            self.num_assignments += 1   # 表示又找到了一个CSP的solution，所以解的个数+1
            new_assignment = {}
            for var in self.csp.variables:
                new_assignment[var] = assignment[var]
            self.all_assignments.append(new_assignment)
            if self.first_assignment_num_operations == 0:
                self.first_assignment_num_operations = self.num_operations
            return

        var = self.get_unassigned_variable(assignment)  # use MCV or not
        ordered_values = self.domains[var]

        if not self.ac3:    # naive backtrace
            # TODO: Problem a
            # TODO: BEGIN_YOUR_CODE
            for value in ordered_values:
                # 检查取值当前value的取值是否合法
                if self.check_factors(assignment=assignment,var=var,val=value):
                    assignment[var] = value
                    self.backtrack(assignment)
                    assignment.pop(var)
            # raise NotImplementedError
            # TODO: END_YOUR_CODE

        else:   # backtrack with arc consistency （AC-3 algorithm）
            # TODO: Problem d
            # TODO: BEGIN_YOUR_CODE
            from copy import deepcopy
            for value in ordered_values:
                # 检查取值当前value的取值是否合法
                if self.check_factors(assignment=assignment,var=var,val=value):
                    assignment[var] = value
                    localCopy = deepcopy(self.domains)
                    self.domains[var] = [value]   # 此处，将var赋值后，var对应的值域domain也就固定且唯一了
                    succeed = self.arc_consistency_check(var)
                    if succeed:
                        # 成功后，再调用backtrack时，self.domains已经被修改，等回溯结束后需要将其复原
                        self.backtrack(assignment)
                    # else: 否则，说明当前var选择该value后，弧相容不能满足，当前value值不能选
                    self.domains = deepcopy(localCopy)
                    assignment.pop(var)


            # raise NotImplementedError
            # TODO: END_YOUR_CODE


    def get_unassigned_variable(self, assignment):
        """Get a currently unassigned variable for a partial assignment.

        If mcv is True, Use heuristic: most constrained variable (MCV)
        Otherwise, select a variable without any heuristics.

        Most Constrained Variable (MCV):
            Select a variable with the least number of remaining domain values.
            Hint: self.domains[var] gives you all the possible values
            Hint: choose the variable with lowest index in self.csp.variables
                for ties


        Args:
            assignment: a dictionary of current assignment.

        Returns
            var: a currently unassigned variable.
        """
        if not self.mcv:
            for var in self.csp.variables:
                if var not in assignment:
                    return var
        else:
            # TODO: Problem c
            # TODO: BEGIN_YOUR_CODE

            best_var = self.csp.variables[0]
            num_of_legalValues_of_best_var = float('inf')
            for var in self.csp.variables:
                if var not in assignment:   # 寻找那些还没被赋值的变量
                    num = 0                 # 统计待赋值变量的合法取值个数
                    for value in self.domains[var]:     # 遍历检查var的每个取值
                        if self.check_factors(assignment=assignment,var=var,val=value): # 说明value是var的一个合法取值
                            num += 1
                    if num < num_of_legalValues_of_best_var:
                        best_var = var
                        num_of_legalValues_of_best_var = num
            return best_var
            # raise NotImplementedError
            # TODO: END_YOUR_CODE

    def arc_consistency_check(self, var):
        """AC-3 algorithm.

        The goal is to reduce the size of the domain values for the unassigned
        variables based on arc consistency.

        Hint: get variables neighboring variable var:
            self.csp.get_neighbor_vars(var)

        Hint: check if a value or two values are inconsistent:
            For unary factors
                self.csp.unaryFactors[var1][val1] == 0
            For binary factors
                self.csp.binaryFactors[var1][var2][val1][val2] == 0

        Args:
            var: the variable whose value has just been set

        Returns
            boolean: succeed or not
        """
        # TODO: Problem d
        # TODO: BEGIN_YOUR_CODE

        arcs_to_be_checked = set()      # the container is a set
        # arcs_to_be_checked = list()   # the container is a queue
        neighbors = self.csp.get_neighbor_vars(var)
        for neighbor in neighbors:
            arcs_to_be_checked.add( (neighbor,var) )
            # arcs_to_be_checked.append( (neighbor,var) )

        while arcs_to_be_checked:  # 当还有弧需要检查时
            deleted = 0     # 默认 没有domain修改发生
            start_var,end_var = arcs_to_be_checked.pop()
            # start_var, end_var = arcs_to_be_checked.pop(0)

            # 对 start_var ——> end_var 进行弧相容检查
            for start_var_value in self.domains[start_var][:]:
                flag = 1  # 默认 当前的start_var_value与end_var的domain有冲突
                for end_var_value in self.domains[end_var]:
                    if self.csp.binary_factors[start_var][end_var][start_var_value][end_var_value]:
                        # 说明当前的start_var_value与end_var的domain相容，flag置为0，表示start_var_value与end_var的domain无冲突，不需要删去
                        flag = 0
                # start_var_value 遍历完end_var的domain后
                if flag:    # 说明当前的start_var_value与end_var的domain有冲突，需要将其删去
                            # 同时，还要将与strat_var的邻居放到集合中去.
                    self.domains[start_var].remove(start_var_value)
                    if len(self.domains[start_var]) == 0: # 说明弧相容检测失败，当前情况无解
                        return False

                    deleted = 1
            # 弧相容检查完毕


            if deleted: # deleted==1，说明start_var的domain有变动，需要将其相关邻居加入进来.
                neighbors = self.csp.get_neighbor_vars(start_var)
                for neighbor in neighbors:
                    if (neighbor , start_var) not in arcs_to_be_checked:    # 如果(neighbor,start_var)原本就存在，那么不需要重复加入
                        arcs_to_be_checked.add((neighbor, start_var))
                    # arcs_to_be_checked.append( (neighbor, start_var) )


        return True
        # raise NotImplementedError
        # TODO: END_YOUR_CODE
