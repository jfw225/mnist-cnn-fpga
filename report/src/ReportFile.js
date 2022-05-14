import ReportFile from "./report-jup.html";
import ReportMD from "./report.md";
import ReportJup from "./report-jup.ipynb";

export default ReportFile;
export const ReportMD_ = ReportMD;

export async function FetchReport(setMDFile) {
    // const file = await fetch(ReportMD);
    // const text = (await file.text()).replaceAll("$$", "\n$$$\n");
    // console.log(text);
    // setMDFile(text);

    const file = await fetch(ReportJup);
    const obj = JSON.parse(await file.text());
    console.log(obj)
    // setMDFile(fff.cells[2].source.join("").replaceAll("$$", "\n$$$\n"));
    // console.log(fff.cells[2].source.join(""));
    // console.log(fff.cells[2].source.join("").replaceAll("$$", "\n$$$\n"))

    const str = obj.cells.reduce((str, cell) => {
        if (cell.cell_type !== "markdown")
            return str;

        return str + cell.source.join("");
    }, "").replaceAll("$$", "\n$$$\n");

    setMDFile(str);


}

