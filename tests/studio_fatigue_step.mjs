// Reads {cases: [{MA, MF, TL, params}]} on stdin and writes the page's exact fatigue step for each case.
import { threeCCrStep } from "../myoInteract/studio/static/sim.js";

let input = "";
for await (const chunk of process.stdin) input += chunk;
const { cases } = JSON.parse(input);
console.log(JSON.stringify(cases.map((c) => threeCCrStep(c.MA, c.MF, c.TL, { ...c.params, restThreshold: c.rest_threshold }))));
