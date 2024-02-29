row = self.Q[(self.Q['Stage'] == stage) & (self.Q['Mission'] == mission)]
msg = f"{row['Name'].iloc[0]}\nRequirement: " + str(row['Requirements'].iloc[0]) + ' | Failable: ' + str(
    row['Failable'].iloc[0]) + ' | Reward: ' + str(row['Rewards'].iloc[0]) + '\nProcedure: ' + str(
    row['Procedure'].iloc[0])
B = HoverButton(self.questDisplay, msg, text=row['Name'].iloc[0],
                disabled=False if (stage == 1) and self.req_met((stage, mission)) else True)
B.mission = mission
B.stage = stage
B.status = 'not started'
datarow.append(B)
self.quests[stage, mission] = B
B.bind(on_press=self.activate)