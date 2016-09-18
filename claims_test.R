library(MASS)
library(xgboost)

set.seed(43)

#---------------------------------------------------------
#
# data prep: Auto claims dataset from MASS
#
#---------------------------------------------------------
N <- nrow(Insurance)
idx <- sample(1:N, floor(0.7*N))
Insurance$Age <- as.numeric(Insurance$Age)
train <- model.matrix(~ District + Group + Age - 1, data=Insurance[idx,])
test <- model.matrix(~ District + Group + Age - 1, data=Insurance[-idx,])

aggregate(Claims/Holders ~ Age, data = Insurance[idx,], mean)

#---------------------------------------------------------
#
# xgboost models
#
#---------------------------------------------------------

dtrain <- xgb.DMatrix(data=train, label=Insurance$Claims[idx])
dtest <- xgb.DMatrix(data=test, label=Insurance$Claims[-idx])

# set # Holders as offset for Possoin count model
setinfo(dtrain, "base_margin", log(Insurance$Holders[idx]))
setinfo(dtest, "base_margin", log(Insurance$Holders[-idx]))

params <- list(max_depth=3, eta=0.1, nthread=4, silent=1, objective='count:poisson')
bst <- xgboost(params=params, data=dtrain, nrounds=100, weight=Insurance$Holders[idx])

params_constrained <- params
params_constrained['monotone_constraints'] = "(0,0,0,0,0,0,0,-1)"
bst_mon <- xgboost(params=params_constrained, data=dtrain, nrounds=100, weight=Insurance$Holders[idx])

preds <- predict(bst, dtest)/Insurance[-idx,'Holders']
aggregate(preds ~ Age, data = cbind(Insurance[-idx,], preds), mean)
plot(as.factor(Insurance[-idx,'Age']), preds)

preds_mon <- predict(bst_mon, dtest)/Insurance[-idx,'Holders']
aggregate(preds_mon ~ Age, data = cbind(Insurance[-idx,], preds_mon), mean)
plot(as.factor(Insurance[-idx,'Age']), preds_mon)

#---------------------------------------------------------
#
# R GBM
#
#---------------------------------------------------------

library(gbm)
rgbm <- gbm(Claims ~ District + Group + Age + offset(log(Holders)),
                distribution="poisson",
                data=Insurance[idx,],
                n.trees=100,
                interaction.depth=4,
                n.minobsinnode=5,
                shrinkage=0.1,
                bag.fraction=0.5,
                train.fraction=1)

rgbm_mon <- gbm(Claims ~ District + Group + Age + offset(log(Holders)),
            distribution="poisson",
            var.monotone=c(0,0,-1),
            data=Insurance[idx,],
            n.trees=100,
            interaction.depth=4,
            n.minobsinnode=5,
            shrinkage=0.1,
            bag.fraction=0.5,
            train.fraction=1)

rpreds <- predict(rgbm, Insurance[-idx,], type='response')
aggregate(rpreds ~ Age, data = cbind(Insurance[-idx,], rpreds), mean)
plot(as.factor(Insurance[-idx,'Age']), rpreds)

rpreds_mon <- predict(rgbm_mon, Insurance[-idx,], type='response')
aggregate(rpreds_mon ~ Age, data = cbind(Insurance[-idx,], rpreds), mean)
plot(as.factor(Insurance[-idx,'Age']), rpreds_mon)

#---------------------------------------------------------
#
# GLM
#
#---------------------------------------------------------

g <- glm("Claims ~ District + Group + Age + offset(log(Holders))",
         data=Insurance[idx,],
         family = "poisson")
summary(g)

gpreds <- predict(g, newdata=Insurance[-idx,], type="response")
plot(as.factor(Insurance[-idx,'Age']), gpreds/Insurance[-idx,'Holders'])
