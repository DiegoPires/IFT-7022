import re
import collections
from utility import get_file_content, get_complet_path

Ingredient = collections.namedtuple('Ingredient', ['item', 'quantity'])

REGEX = r"""
(([0-99])+(\,\d{0,})?(\/[0-9])?(\snoix?)?(\senveloppe?)?(\tasse?)?(\smorceau?)?(\strait?)?)(\ ?(gousses|(B|b)ouquet|(R|r)ondelle|feuilles|tasse(s|)|cuillÃ¨re(s|)((\sÃ(\s*)cafÃ©)|(\sÃ(\s*)soupe))?|m(L|l)|(g\s)|l(b|B)|tranches|pintes|gallon|pincÃ©e|cl|oz|(c\.\s{0,}Ã\s{0,}((s\.)|(c\.)|(\.c)|(\.s)|thÃ©|soupe))))?|((.*)\)|(U|u)ne pincÃ©e|(A|a)u goÃ»t|(Q|q)uelques|(F|f)euilles)
"""

REGEX_TEXT = r"^(de |d'|dâ€™|du|des)"

pattern = re.compile(REGEX, re.VERBOSE | re.IGNORECASE)
item_pattern = re.compile(REGEX_TEXT, re.VERBOSE | re.IGNORECASE)

def get_ingredient(text):
    result = pattern.match(text)
    if (result != None):
        item = text[result.end():].strip()
        # 1_improvement - Before this little plus, we were at 20% succes rate
        result_item_pattern = item_pattern.match(item)
        if (result_item_pattern != None):
            item = item[result_item_pattern.end():].strip()

        return Ingredient(fix_ingredient(item), result.group().strip())
    else:
        return Ingredient(fix_ingredient(text), "")

def fix_ingredient(item):
    return item.replace(
            ", pour dÃ©corer","").replace(
                "en purÃ©e", "").replace(
                    "rÃ¢pÃ©", "").replace(
                        " Ã©crasÃ©s", "").replace(
                            ", tranchÃ©","").replace(
                                "Ã©mincÃ©", "").replace(
                                    "au goÃ»t", "").replace(
                                        "(surtout pas en boÃ®te)", "").replace(
                                            " pelÃ©es, en tranches", "").replace(
                                                " battu", "").replace(
                                                    "tranchÃ©", "").replace(
                                                        "hachÃ©es", ""
                                    ).strip()

def main():
    ingredients_text = get_file_content(get_complet_path("ingredients.txt"))
    solutions_text = get_file_content(get_complet_path("ingredients_solutions.txt"))

    quantity_correct = 0
    quantity_wrong = 0

    print("###################### SOLUTION")

    for count,ingredient_text in ingredients_text.items():
        ingredient = get_ingredient(ingredient_text)
        ingredient_solution = solutions_text[count].split("   ")
        solution_quantity = ingredient_solution[1].replace("QUANTITE:","")
        solution_item = ingredient_solution[2].replace("INGREDIENT:","")

        if (solution_quantity == ingredient.quantity and solution_item == ingredient.item):
            quantity_correct +=1
        else:
            quantity_wrong +=1
            print("NOT MATCH:\n {!r} \n Solution  (item='{}', quantity='{}')\n".format(ingredient, solution_item,solution_quantity))

    succes_rate = quantity_correct * 100 / len(ingredients_text)
    print("\nResult: CORRECT({}) x WRONG({}) - {:.2f}%".format(quantity_correct, quantity_wrong, succes_rate))
    print()

if __name__ == '__main__':  
   main()