/** Regex used to determine the language of the code block. */
const LANG_RE = /^#*\s*(\w+)\n/;

export default async function FetchReport(setMDFile) {
    const ReportJup = require("./report-jup.ipynb");
    const file = await fetch(ReportJup);
    const obj = JSON.parse(await file.text());

    const str = obj.cells.reduce((str, cell) => {
        if (cell.cell_type === "markdown") {

            return str + cell.source.join("");
        } else if (cell.cell_type === "code") {
            let match = cell.source[0].match(LANG_RE);

            if (!match)
                throw Error(`${cell}`);

            return (str + "\n```" + `${match[1]}\n` +
                cell.source.slice(1).join("") + "\n```\n");
        } else
            throw Error(`${cell}`);

    }, "").replaceAll("$$", "\n$$$\n");

    setMDFile(str);


}

