export default async function FetchReport(setMDFile) {
    const ReportJup = require("./report-jup.ipynb");
    const file = await fetch(ReportJup);
    const obj = JSON.parse(await file.text());

    const str = obj.cells.reduce((str, cell) => {
        if (cell.cell_type !== "markdown")
            return str;

        return str + cell.source.join("");
    }, "").replaceAll("$$", "\n$$$\n");

    setMDFile(str);


}

