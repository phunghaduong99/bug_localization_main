/*******************************************************************************
 * Copyright (c) 2010 IBM Corporation and others.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     IBM Corporation - initial API and implementation
 ******************************************************************************/

package org.eclipse.e4.ui.tests.reconciler;

import java.util.Collection;
import org.eclipse.e4.ui.model.application.MApplication;
import org.eclipse.e4.ui.model.application.ui.menu.MMenuContribution;
import org.eclipse.e4.ui.model.application.ui.menu.impl.MenuFactoryImpl;
import org.eclipse.e4.ui.workbench.modeling.ModelDelta;
import org.eclipse.e4.ui.workbench.modeling.ModelReconciler;

public abstract class ModelReconcilerMenuContributionTest extends
		ModelReconcilerTest {

	private void testMenuContribution_PositionInParent(String before,
			String after) {
		MApplication application = createApplication();

		MMenuContribution contribution = MenuFactoryImpl.eINSTANCE
				.createMenuContribution();
		application.getMenuContributions().add(contribution);
		contribution.setPositionInParent(before);

		saveModel();

		ModelReconciler reconciler = createModelReconciler();
		reconciler.recordChanges(application);

		contribution.setPositionInParent(after);

		Object state = reconciler.serialize();

		application = createApplication();
		contribution = application.getMenuContributions().get(0);

		Collection<ModelDelta> deltas = constructDeltas(application, state);

		assertEquals(1, application.getMenuContributions().size());
		assertEquals(contribution, application.getMenuContributions().get(0));
		assertEquals(before, contribution.getPositionInParent());

		applyAll(deltas);

		assertEquals(1, application.getMenuContributions().size());
		assertEquals(contribution, application.getMenuContributions().get(0));
		assertEquals(after, contribution.getPositionInParent());
	}

	public void testMenuContribution_PositionInParent_EmptyEmpty() {
		testMenuContribution_PositionInParent("", "");
	}

	public void testMenuContribution_PositionInParent_EmptyString() {
		testMenuContribution_PositionInParent("", "id");
	}

	public void testMenuContribution_PositionInParent_StringEmpty() {
		testMenuContribution_PositionInParent("id", "");
	}

	public void testMenuContribution_PositionInParent_StringStringUnchanged() {
		testMenuContribution_PositionInParent("id", "id");
	}

	public void testMenuContribution_PositionInParent_StringStringChanged() {
		testMenuContribution_PositionInParent("id", "id2");
	}

	private void testMenuContribution_ParentID(String before, String after) {
		MApplication application = createApplication();

		MMenuContribution contribution = MenuFactoryImpl.eINSTANCE
				.createMenuContribution();
		application.getMenuContributions().add(contribution);
		contribution.setParentId(before);

		saveModel();

		ModelReconciler reconciler = createModelReconciler();
		reconciler.recordChanges(application);

		contribution.setParentId(after);

		Object state = reconciler.serialize();

		application = createApplication();
		contribution = application.getMenuContributions().get(0);

		Collection<ModelDelta> deltas = constructDeltas(application, state);

		assertEquals(1, application.getMenuContributions().size());
		assertEquals(contribution, application.getMenuContributions().get(0));
		assertEquals(before, contribution.getParentId());

		applyAll(deltas);

		assertEquals(1, application.getMenuContributions().size());
		assertEquals(contribution, application.getMenuContributions().get(0));
		assertEquals(after, contribution.getParentId());
	}

	public void testMenuContribution_ParentID_NullNull() {
		testMenuContribution_ParentID(null, null);
	}

	public void testMenuContribution_ParentID_NullEmpty() {
		testMenuContribution_ParentID(null, "");
	}

	public void testMenuContribution_ParentID_NullString() {
		testMenuContribution_ParentID(null, "id");
	}

	public void testMenuContribution_ParentID_EmptyNull() {
		testMenuContribution_ParentID("", null);
	}

	public void testMenuContribution_ParentID_EmptyEmpty() {
		testMenuContribution_ParentID("", "");
	}

	public void testMenuContribution_ParentID_EmptyString() {
		testMenuContribution_ParentID("", "id");
	}

	public void testMenuContribution_ParentID_StringNull() {
		testMenuContribution_ParentID("id", null);
	}

	public void testMenuContribution_ParentID_StringEmpty() {
		testMenuContribution_ParentID("id", "");
	}

	public void testMenuContribution_ParentID_StringStringUnchanged() {
		testMenuContribution_ParentID("id", "id");
	}

	public void testMenuContribution_ParentID_StringStringChanged() {
		testMenuContribution_ParentID("id", "id2");
	}
}
