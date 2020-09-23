# Dobókockák alakjának meghatározása diszkrét eloszlásokhoz

*BSc szakdolgozat*

A szerepjátékokban gyakran használnak különféle számú lappal rendelkező dobókockákat a hagyományos (valóban kockának nevezhető) 6 oldalú kockák mellett. Jellemzően ezek olyanok, hogy a kocka azonos valószínűséggel áll meg bármelyik lapján, és a felső lapon lévő jelölés jelzi, hogy mennyi a dobott érték. A dolgozat ezt a problémát igyekszik általánosítani olyan módon, hogy egy rögzített diszkrét eloszláshoz igyekszik megtalálni azt a térbeli alakzatot, amelynél a dobott értékek előfordulása az eloszlást követi.

A probléma megoldásához különféle módszereket vizsgál a dolgozat. Elméleti úton elemzi azt, hogy az alakzatokat hogyan érdemes leírni, az eloszlás alapján hogyan lehet azokat közvetlenül meghatározni. A kockadobást, mint szilárdtest mechanikai problémát szimulációk segítségével tárgyalja. Bemutatja, hogy egy-egy eloszláshoz milyen alternatív alakzatok jöhetnek szóba, például az optimalizálás feltételeinek megváltoztatásával.

## Észrevételek, ötletek

* Optimalizálási feltétel lehet a lapok területének egyezősége, vagy a csúcsok gömbfelületre illeszkedése.
* Érdemes azt is megnézni, hogy a szabályos testeket is ki tudja-e számolni a program, az mennyire sokáig tart neki.
* Bizonyos esetekben a csúcsok is jelölhetik a dobott értéket.
* Dobásnál figyelni kell arra, hogy a két szöggel megadott gömbi koordinátarendszerben a pontok nem egyenletesen helyezkednek el a felületen. Valamilyen tesszellációs módszert is érdemes lehet használni.

