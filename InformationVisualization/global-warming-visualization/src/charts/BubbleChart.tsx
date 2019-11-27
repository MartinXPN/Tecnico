import * as d3 from "d3";
import ScatterPlot, {Props} from "./ScatterPlot";
import {GdpTemperatureMeatGhgData} from "../entities";


export default class BubbleChart extends ScatterPlot {

    constructor(props: Props) {
        super(props);

        this.xLabel = 'GDP per-capita';
        this.yLabel = 'Meat consumption per-capita';
        this.title = 'Meat consumption GDP and GHG emissions';
    }

    getX = (d: GdpTemperatureMeatGhgData) => d.gdp;
    getY = (d: GdpTemperatureMeatGhgData) => d.meat_consumption;


    handleCountryYear = (svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
                         dataPoint: (GdpTemperatureMeatGhgData | undefined),
                         country: string, identifier: string,
                         color: string, h: number) => {
        if (!dataPoint) {
            svg.select(`circle[title='${identifier}-${country}']`).attr('visibility', 'hidden');
            return;
        }

        svg.select(`circle[title='${identifier}-${dataPoint.country}']`)
            .on("mouseover", () => {
                this.props.hoverCountry(country);
                this.tooltip.style("visibility", "visible");
                this.tooltip.html(`<div><strong>${dataPoint.country}</strong></div>${Math.round(dataPoint.ghg_emission / 1000) + 'K'} greenhouse gas emissions<div>${dataPoint.meat_consumption} meat consumed per-capita</div>${dataPoint.gdp} GDP per-capita`);
            })
            .on("mousemove", () => this.tooltip.style("top", (d3.event.pageY - 10) + "px").style("left", (d3.event.pageX + 10) + "px"))
            .on("mouseout", () => {
                this.props.hoverCountry(undefined);
                this.tooltip.style("visibility", "hidden");
            })
            .on("click", () => this.props.selectCountry(country))
            .transition().duration(250)
            .attr('cx', this.xScale(dataPoint.gdp))
            .attr('cy', h - this.yScale(dataPoint.meat_consumption))
            .attr('r', Math.log(dataPoint.ghg_emission))
            .attr("fill", color)
            .attr('visibility', 'visible')
            .attr('opacity', this.props.currentHoveredCountry === country ? 1 : 0.8);
    };
}
