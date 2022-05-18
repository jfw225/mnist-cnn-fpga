import MarkdownRender from './MarkdownRender';

/** Mapping of images to make sure they properly load at runtime. */
const IMAGE_MAP = {
    "./images/tf.png": require("./images/tf.png"),
    "./images/conv.png": require("./images/conv.png"),
    "./images/sigmoid.png": require("./images/sigmoid.png"),
    "./images/mp.png": require("./images/mp.png"),
    "./images/fullyconnect.png": require("./images/fullyconnect.png")
};

/** The regex used to match container comments. */
const CONTAINER_RE = /^<!-- container:\s*(\w*)\s*-->/;

/** Object used to match strings to their proper formatters. */
const MATCHERS = [
    {
        re: /^#{1}\s*((?:\w|\s|\/)+)$/,
        elem: (match: RegExpMatchArray) => <h1 className="w3-wide"><b>{match[1]}</b></h1>
    },
    {
        re: /^#{2}\s*((?:\w|\s|\/)+)$/,
        elem: (match: RegExpMatchArray) => <h2 className="w3-justify"><u>{match[1]}</u></h2>
    },
    {
        re: /^#{4}\s*((?:\w|\s|:|\/)+)$/,
        elem: (match: RegExpMatchArray) => <h4 className="w3-wide"><b><u>{match[1]}</u></b></h4>
    },
    {
        re: /^#{5}\s*((?:\w|\s|:|\/)+)$/,
        elem: (match: RegExpMatchArray) => <h4 className="w3-justify"><b>{match[1]}</b></h4>
    },
    {
        re: /^!\[(?:\w|\s)*\]\((.*)\)$/,
        elem: (match: RegExpMatchArray) => {
            if (!IMAGE_MAP.hasOwnProperty(match[1]))
                throw new Error(`Must add ${match[1]} to the image map!`)

            // @ts-ignore
            return <img className="w3-justify" src={IMAGE_MAP[match[1]]} />;
        }
    },
    {
        re: /^!\[video\]\[((?:\w|\s)+)\]\((.*)\)$/,
        elem: (match: RegExpMatchArray) => {

            return <iframe width="560" height="315" src={match[2].replace("watch?v=", "embed/")} title={match[1]} allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"></iframe>;
        }

    }
]

type ContainerElement = JSX.Element | Array<JSX.Element>;

/** The container keys. */
enum CONTAINER_KEY {
    DEFAULT = "default",
    DARK = "dark"
};


/** Dictionary of container creators. */
const CONTAINER_FNS = {
    [CONTAINER_KEY.DEFAULT]: (e: ContainerElement) => (
        <div className="w3-white">
            <div className="w3-container w3-content w3-center w3-padding-64" style={{ maxWidth: 800 }} id="report">
                {e}
            </div>
        </div>
    ),
    [CONTAINER_KEY.DARK]: (e: ContainerElement) => (
        <div className="w3-black">
            <div className="w3-container w3-content w3-center w3-padding-64" style={{ maxWidth: 800 }} id="report">
                {e}
            </div>
        </div>
    )
}

function createContainer(key: CONTAINER_KEY, e: ContainerElement): JSX.Element {
    if (!key || !CONTAINER_FNS.hasOwnProperty(key))
        key = CONTAINER_KEY.DEFAULT;

    return CONTAINER_FNS[key](e);
}

function strToElem(str: string) {
    for (const { re, elem } of MATCHERS) {
        // console.log(str, re.exec(str));
        let match = str.match(re);
        if (match)

            return elem(match);
    }

    return str;
}



export function ParseMD(mdBody: string): ContainerElement {
    let mdSplit = mdBody.split("\n");
    // console.log(mdSplit);

    /* The key used to find the container function. */
    let containerKey: CONTAINER_KEY = CONTAINER_KEY.DEFAULT;

    let containers: Array<JSX.Element> = [];

    /* Array to keep track of elements to be placed into a container. */
    let elements: Array<JSX.Element> = [];

    /* String to keep track of overflow. */
    let str = "\n";

    /* Function used to convert `str` to JSX element and add to `elements`. */
    const handleStr = () => {
        if (str !== "\n") {
            elements.push(
                <div className="w3-justify">
                    <MarkdownRender>{str}</MarkdownRender>
                </div>);
            str = "\n";
        }
    };

    /* Function used to convert `elements` to container and add to `containers`. */
    const handleElements = () => {
        if (elements.length) {
            containers.push(createContainer(containerKey, elements));
            elements = [];
        }
    };

    mdSplit.map(strToElem).forEach((v) => {
        // console.log(v);
        if (typeof (v) !== "string") {

            handleStr();

            return elements.push(v);
        }

        let match = v.match(CONTAINER_RE);
        if (match) {
            handleStr();
            handleElements();

            containerKey = match[1] as CONTAINER_KEY;


        } else
            str += v + "\n";
    })

    // handle the string
    handleStr();

    // create last container
    handleElements();

    return containers;
}