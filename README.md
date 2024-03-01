# article-deconstructor

Academic Article Reader/Deconstructor



Goal: Program that can take the text of any article, determine its complexity, and provide simpler reading suggestions for specific concepts in the text. Mostly intended to be used for political/critical theory. Ex: imagine an article about the overlap between Marxism and contemporary theories of statehood. The program would point to simpler articles/resources that explain these concepts independently. Of course, with most examples, it would be much more complicated than having two concepts.

Thoughts:
- Set of associated concepts/fields of thought which all independently have associated keywords, phrases, and simpler explanatory texts. This might seem daunting, but we could plausibly work with an existing “almanac” for political theory or something, or alternatively restrict the scope of the project to be something more manageable. I don’t want to have to make this database, but enough resources like encyclopedias or almanacs exist that I think it’s probably plausible.
- Each imported text would be converted into a table of integers which all demonstrate the presence of EACH known concept in the imported text. AI would probably be used, but the integer number could plausibly also be the amount of “associated concept phrases” relative to the length of the article. Ex: if 200 out of 900 of the words are talking about liberalism, then that has some associated value.
Output would look something like: here is how complex your text is, the most prominent concepts are (x) with a prominence score of ??,  y with a prominence score of ?? … for x, we recommend looking at ??, for y, we recommend looking at ??

