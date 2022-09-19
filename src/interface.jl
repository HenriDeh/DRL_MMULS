export to_instance
function to_instance(is::InventorySystem, gamma)
    @assert only(is.bom) isa EndProduct
    @assert only(only(is.bom).sources) isa Supplier
    ep = only(is.bom)
    inv = ep.inventory
    supplier = only(ep.sources)
    market = ep.market
    leadtime = supplier.leadtime.leadtime
    Scarf.Instance(inv.holding_cost.h, market.stockout_cost.b, supplier.order_cost.K, 
        supplier.order_cost.c, cv(market.demand_dist), leadtime, rand.(market.forecast_reset[1].itr.xs), gamma, 
        backlog = !market.lostsales)
end

function to_instance(is::InventorySystem, gamma, range)
    @assert only(is.bom) isa EndProduct
    @assert only(only(is.bom).sources) isa Supplier
    ep = only(is.bom)
    inv = ep.inventory
    supplier = only(ep.sources)
    market = ep.market
    leadtime = supplier.leadtime.leadtime
    Scarf.Instance(inv.holding_cost.h, market.stockout_cost.b, supplier.order_cost.K, 
        supplier.order_cost.c, cv(market.demand_dist), leadtime, rand.(market.forecast_reset[1].itr.xs)[range], gamma, 
        backlog = !market.lostsales)
end