import * as FileSystem from "LensStudio:FileSystem";
const relativeJsonSourcePath = '../Data/ls-api-definition.json';
class DefinitionParser {
    constructor(fc) {
        this.fc = fc;
        this.apiInfo = JSON.parse(fc);
    }
    fuzzyMatch(input, comparedTo) {
        if (!input || !comparedTo) {
            return false;
        }
        const normalizedInput = input.split(".")[0].toLowerCase().replace(/\s/g, '');
        const normalizedComparedTo = comparedTo.toLowerCase().replace(/\s/g, '');
        if (normalizedComparedTo.length === 0) {
            return true;
        }
        let comparedToIndex = 0;
        let inputIndex = 0;
        let matchCount = 0;
        while (comparedToIndex < normalizedComparedTo.length && inputIndex < normalizedInput.length) {
            if (normalizedComparedTo[comparedToIndex] === normalizedInput[inputIndex]) {
                matchCount++;
                comparedToIndex++;
                inputIndex++;
            }
            else {
                comparedToIndex++;
            }
        }
        const longerLength = Math.max(normalizedInput.length, normalizedComparedTo.length);
        const allowedMissing = Math.floor(longerLength * 0.5);
        return (longerLength - matchCount) <= allowedMissing;
    }
    getDefinition(nodeNameToFind) {
        const potentialMatches = [];
        for (var type in this.apiInfo) {
            for (var nodeName in this.apiInfo[type]) {
                if (this.fuzzyMatch(nodeNameToFind, nodeName)) {
                    potentialMatches.push(this.apiInfo[type][nodeName].text);
                }
            }
        }
        return JSON.stringify(potentialMatches);
    }
}
const dtsFilePath = new Editor.Path(import.meta.resolve(relativeJsonSourcePath));
const fileContent = FileSystem.readFile(dtsFilePath);
const lsDefinitionParser = new DefinitionParser(fileContent);
export function getDefinition(nodeNameToFind) {
    try {
        const extractedDtsContent = lsDefinitionParser.getDefinition(nodeNameToFind);
        return extractedDtsContent;
    }
    catch (error) {
        console.error('Error getting definition:', error);
        return null;
    }
}
