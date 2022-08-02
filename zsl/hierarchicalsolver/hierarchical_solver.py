from curses import KEY_A1
import numpy as np

import community.community_louvain as community

from Orange.clustering.hierarchical import data_clustering, WEIGHTED
from Orange.data import Table, Domain
from Orange.data.variable import StringVariable
from Orange.distance.distance import Cosine
from Orange.widgets.unsupervised.owhierarchicalclustering import clusters_at_height

from typing import Tuple, Dict
from itertools import chain
from collections import Counter

class HiCA:

    GROUP_BY            = "first superclass"
    MYSTERY             = "TOGUESS"
    KEY                 = "embeddings"
    CLUSTER_THRESOLD    = 0.85
    SIM_THRESOLD        = 0.3

    @staticmethod
    def left_join(cls, complete_table, supp_info_table, key: str = "embeddings") -> Table:
        """add all metas column from supp_info_table to complete_table using key as joint
        """
        assert key in list(map(lambda x : x.name, supp_info_table.domain.metas)), "embeddings name not present in additional data"

        name_supp_data = [i.name for i in chain(supp_info_table.domain.metas, 
                                                supp_info_table.domain.variables, 
                                                supp_info_table.attributes) if i.name != key]
                                                
        supp_list_list = [[] for i in range(len(name_supp_data))]

        for s in complete_table:
            done = False
            for d in supp_info_table:
                if s[key] == d[key]:
                    for i, name in enumerate(name_supp_data):
                        supp_list_list[i].append(d[name])
                    done = True
                    break
            if not done:
                for i, name in enumerate(name_supp_data):
                    supp_list_list[i].append("?")

        for i, name in enumerate(name_supp_data):
            # print(f"adding {name}")
            complete_table = complete_table.add_column(StringVariable(name), supp_list_list[i])

        return complete_table

    def __init__(self, mystery_embedding, prior_knowledge_table, superclass_embeddings, standardize_mystery : bool = False) -> None:
        self.mystery_embedding      = mystery_embedding
        self.prior_knowledge_table  = prior_knowledge_table
        self.standardized_myster    = standardize_mystery
        self.superclass_embeddings  = superclass_embeddings

        self.mystery_table = Table.from_numpy(self.prior_knowledge_table.domain, [np.array(self.mystery_embedding)], 
                                                                        Y = None, 
                                                                        metas = np.char.asarray([[HiCA.MYSTERY, "?", "?"]]))
        if standardize_mystery:
            toguess_table = self.standardize_first(self.mystery_table)
            

        self.table = Table.concatenate([self.prior_knowledge_table, self.mystery_table])

        for i in self.table[-1::-1]:
            if i[HiCA.KEY] == HiCA.MYSTERY:
                self.mystery_index = self.table.index(i)
                break

    def __standardize_first(self, table):
        values = table[0]
        mean = np.mean(values)
        std  = np.std(values)

        for v in range(len(values)):
            values[v] = (values[v] - mean) / std

        return Table.from_numpy(table.domain, [values], None, table.metas)

    def __parent_of_mystery(self, cluster):
        res = None
        for branch in cluster.branches:
            if branch.is_leaf:
                if branch.value.index == self.mystery_index:
                    return cluster
            else: 
                res = self.parent_of_mystery(branch)
                if res is not None:
                    return res
        
    def __first_child(self, root):
        if root.is_leaf:
            return root
        else:
            return self.first_child(root.branches[0])


    def __closest_to(self, cluster):
        if len(cluster.branches) == 1:
            return None

        next = False
        for i, branch in enumerate(cluster.branches):
            if next:
                return self.__first_child(branch)

            if branch.is_leaf:
                if branch.value.index == self.mystery_index:
                    if i == 0:
                        next = True
                    else:
                        return self.__first_child(cluster.branches[i-1])

    def __add_to_list(self, cluster, list_to_add_to):
        """ decompose a cluster tree by adding the index of all children in the list
        """
        if cluster.is_leaf:
            list_to_add_to.append(cluster.value.index)

        for i, branch in enumerate(cluster.branches):
            self.__add_to_list(branch, list_to_add_to)

    def __clusterize(self, thresold = None) -> Table:
        """clusterize a Oranga Table based on the height of THRESOLD
        """

        root = data_clustering(self.table, distance=Cosine, linkage=WEIGHTED)
        parent_cluster = self.__parent_of_mystery(root)
        if thresold is None:
            thresold = min(parent_cluster.value.height + 0.05, 1)

        cluster_tree = clusters_at_height(root, thresold)

        list_cluster = {}
        closest = None
        mystery_len_cluster = -1
        for i, cluster in enumerate(cluster_tree):
            cluster_name     = 'C' + str(i) 

            current = []
            self.__add_to_list(cluster, current)
            if self.mystery_index in current: 
                mystery_len_cluster = len(current)
                closest = self.__closest_to(parent_cluster)

            for item_index in current:
                list_cluster[item_index] = cluster_name

        table = self.table.add_column(StringVariable("Cluster"), [list_cluster[i] for i in range(len(self.table))])

        return table, closest.value.index, thresold, i

    def __compute(self, lst):
        counter = Counter(lst)
        return counter.most_common(len(lst))

    def __one_pass(self, cluster_thresold : float, sim_thresold : float, keep_cluster_line : bool = False) -> Tuple[Table, Dict]:
        assert HiCA.GROUP_BY in list(map(lambda x: x.name, chain(self.table.domain.metas, 
                                                            self.table.domain.variables, 
                                                            self.table.domain.attributes))), "Group by not in the Table !"
        
        format = (f"{len(self.table)}x{len(self.table.domain.attributes)}")
        table, closest, thresold, nb_cluster = self.__clusterize(cluster_thresold)

        # Cluster split
        toguess_cluster = [d["Cluster"] for d in table if d[HiCA.KEY] == HiCA.MYSTERY][0]
        in_cluster_table  = Table.from_list(table.domain, [d for d in table if d["Cluster"].value == toguess_cluster])
        out_cluster_table = Table.from_list(table.domain, [d for d in table if d["Cluster"].value != toguess_cluster])
        if len(in_cluster_table) == 0 : return [], "cluster is empty"


        # Group by computation
        main_superclass_count_list = self.__compute([row[HiCA.GROUP_BY].value for row in in_cluster_table])
        # equality case with "?", take the second
        ind = 1 if main_superclass_count_list[0][0] == "?" and len(main_superclass_count_list) > 1 else 0
        main_superclass = main_superclass_count_list[ind][0]

        main_superclass_table = Table.from_list(self.superclass_embeddings.domain, 
                                                [i for i in self.superclass_embeddings if i[HiCA.KEY] == main_superclass])
        main_superclass_table = Table.concatenate([ in_cluster_table, 
                                                    Table.from_table(out_cluster_table.domain, main_superclass_table)])


        # thresold computation
        to_copy_row_instance = [d for d in main_superclass_table if d[HiCA.KEY] == HiCA.MYSTERY][0]
        to_copy = list(to_copy_row_instance.attributes())

        to_compare_row_instance = [d for d in main_superclass_table if d["Cluster"] == "?"][0]
        to_compare = list(to_compare_row_instance.attributes())

        dead_row = [k for k, (i, j) in enumerate(zip(to_copy, to_compare)) if abs(i - j) <= sim_thresold]

        # reconstruct the table filtering dead row and cluster. Remove used cluster row if 
        # keep_cluster_line is set to False
        new_domain = Domain(attributes = [i for i in out_cluster_table.domain.attributes if int(i.name) not in dead_row], 
                            metas      = [i for i in out_cluster_table.domain.metas if i.name != "Cluster"])

        # do the same on the data
        data_attr, data_meta = [], []
        whole_data = list(out_cluster_table) + list(self.mystery_embedding)
        if keep_cluster_line: whole_data += list(in_cluster_table)

        for rowinstance in whole_data:
            data_attr.append([rowinstance[k] for k, i in enumerate(out_cluster_table.domain.attributes) if int(i.name) not in dead_row])
            data_meta.append([rowinstance.metas[k] for k, i in enumerate(out_cluster_table.domain.metas) if i.name != "Cluster"])

        return Table.from_numpy(new_domain, X = data_attr, metas = data_meta), \
            {"cluster_name" : main_superclass,
                "cluster_size" : len(in_cluster_table) - 1,
                "cluster_thresold": thresold,
                "closest_to_myster" : table[closest]["embeddings"].value if closest is not True else None,
                "number_of_cluster" : nb_cluster,
                "format_at_beginning": format,
                "keep_cluster_line"  : keep_cluster_line,
                "sim_thresold"       : sim_thresold,
                "removed_col"        : len(dead_row) 
            }

    def solve(self, cluster_thresold_lambda, sim_thresold_lambda):

        advancement = []
        for i in range(5):
            table, data = self.__one_pass(  cluster_thresold  = cluster_thresold_lambda(i), 
                                            sim_thresold      = sim_thresold_lambda(i),
                                            keep_cluster_line = False) 
            advancement.append(data)

            if len(table) <= 1:
                break
        return advancement
