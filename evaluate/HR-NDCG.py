 def evaluate(self,topK):
        def getHitRatio(ranklist, targetItem):
            for item in ranklist:
                if item == targetItem:
                    return 1
            return 0
        def getNDCG(ranklist, targetItem):
            for i in range(len(ranklist)):
                item = ranklist[i]
                if item == targetItem:
                    return math.log(2) / math.log(i+2)
            return 0

        hr = []
        NDCG = []
        testUser = self.testUser
        testItem = self.testItem
        testRate = self.predict(self.xuser_test, self.xitem_test).reshape(-1,100)
        for i in range(len(testUser)):
            target = testItem[i][0]
            item_score_dict = {}
            for j in range(len(testItem[i])):
                item=testItem[i][j]
                item_score_dict[item]=testRate[i][j]

            ranklist = heapq.nlargest(topK, item_score_dict, key=item_score_dict.get)

            tmp_hr = getHitRatio(ranklist, target)
            tmp_NDCG = getNDCG(ranklist, target)
            hr.append(tmp_hr)
            NDCG.append(tmp_NDCG)
        return np.mean(hr), np.mean(NDCG)
